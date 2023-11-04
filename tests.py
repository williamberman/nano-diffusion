import dataclasses
import os
import tempfile
from collections import namedtuple

import numpy as np
import scipy
import torch
import torch.distributed as dist
from diffusers import (AutoencoderKL, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from huggingface_hub import hf_hub_download
from PIL import Image

import train
from data import get_controlnet_inpainting_conditioning_image
from diffusion import (heun_ode_solver, make_sigmas, rk4_ode_solver,
                       sdxl_diffusion_loop, sdxl_text_conditioning,
                       set_with_tqdm)
from models import (SDXLAdapter, SDXLCLIPOne, SDXLCLIPTwo, SDXLControlNet,
                    SDXLUNet, SDXLVae, make_clip_tokenizer_one_from_hub,
                    make_clip_tokenizer_two_from_hub,
                    set_attention_implementation)

set_attention_implementation("torch_2.0_scaled_dot_product")

set_with_tqdm(True)


class InferenceTests:
    def __init__(self, device, dtype):
        self.tokenizer_one = make_clip_tokenizer_one_from_hub()
        self.tokenizer_two = make_clip_tokenizer_two_from_hub()
        self.device = device
        self.dtype = dtype

    def vae(self):
        vae = SDXLVae.load_fp16_fix(device=self.device)
        vae.to(self.dtype)
        return vae

    def vae_(self):
        vae_ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype)
        vae_.to(device=self.device)
        return vae_

    def unet(self):
        if self.dtype == torch.float32:
            unet = SDXLUNet.load_fp32(device=self.device)
        elif self.dtype == torch.float16:
            unet = SDXLUNet.load_fp16(device=self.device)
        else:
            assert False

        return unet

    def unet_(self):
        if self.dtype == torch.float32:
            unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
            unet_.to(device=self.device)
        elif self.dtype == torch.float16:
            unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16, variant="fp16")
            unet_.to(device=self.device)
        else:
            assert False

        return unet_

    def text_encoder_one(self):
        if self.dtype == torch.float32:
            text_encoder_one = SDXLCLIPOne.load_fp32(device=self.device)
        elif self.dtype == torch.float16:
            text_encoder_one = SDXLCLIPOne.load_fp16(device=self.device)
        else:
            assert False

        return text_encoder_one

    def text_encoder_two(self):
        if self.dtype == torch.float32:
            text_encoder_two = SDXLCLIPTwo.load_fp32(device=self.device)
        elif self.dtype == torch.float16:
            text_encoder_two = SDXLCLIPTwo.load_fp16(device=self.device)
        else:
            assert False

        return text_encoder_two

    def sdxl_pipe(self):
        if self.dtype == torch.float32:
            sdxl_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
            sdxl_pipe.set_progress_bar_config(disable=True)
        elif self.dtype == torch.float16:
            sdxl_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16)
            sdxl_pipe.set_progress_bar_config(disable=True)
        else:
            assert False

        sdxl_pipe.enable_model_cpu_offload()

        return sdxl_pipe

    @torch.no_grad()
    @torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    def test_sdxl_vae(self):
        print("test_sdxl_vae")

        vae = self.vae()

        image = Image.open(hf_hub_download("williamberman/misc", "two_birds_on_branch.png", repo_type="dataset"))
        image = image.convert("RGB")
        image = image.resize((1024, 1024))
        image = vae.input_pil_to_tensor(image)
        image = image.to(dtype=vae.dtype, device=vae.device)

        encoder_output = vae.encode(image, generator=torch.Generator(self.device).manual_seed(0))
        decoder_output = vae.decode(encoder_output)

        del vae

        vae_ = self.vae_()

        expected_encoder_output = vae_.encode(image).latent_dist.sample(generator=torch.Generator(self.device).manual_seed(0))
        expected_encoder_output = expected_encoder_output * vae_.config.scaling_factor
        expected_decoder_output = vae_.decode(expected_encoder_output / vae_.config.scaling_factor).sample

        del vae_

        total_diff = (expected_encoder_output.float() - encoder_output.float()).abs().sum()
        assert total_diff == 0

        total_diff = (expected_decoder_output.float() - decoder_output.float()).abs().sum()
        assert total_diff == 0

    @torch.no_grad()
    @torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    def test_sdxl_unet(self):
        print("test_sdxl_unet")

        prompts = ["horse"]

        text_encoder_one = self.text_encoder_one()
        text_encoder_two = self.text_encoder_two()
        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_two.device),
        )

        del text_encoder_one, text_encoder_two

        unet = self.unet()

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

        micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet.device)

        x_t = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet.dtype, device=unet.device, generator=torch.Generator(self.device).manual_seed(0))

        t = torch.tensor([500], dtype=torch.long, device=self.device)

        unet_output = unet(
            x_t=x_t,
            t=t,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            micro_conditioning=micro_conditioning,
        )

        del unet

        unet_ = self.unet_()

        x_t = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet_.dtype, device=unet_.device, generator=torch.Generator(self.device).manual_seed(0))

        t = torch.tensor([500], dtype=torch.long, device=self.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet_.dtype, device=unet_.device)

        micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet_.device)

        expected_unet_output = unet_(
            sample=x_t, timestep=t, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs={"time_ids": micro_conditioning, "text_embeds": pooled_encoder_hidden_states}
        ).sample

        del unet_

        total_diff = (expected_unet_output.float() - unet_output.float()).abs().sum()

        assert total_diff == 0

    @torch.no_grad()
    @torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    def test_text_to_image(self):
        print("test_text_to_image")

        text_encoder_one = self.text_encoder_one()
        text_encoder_two = self.text_encoder_two()

        prompts = ["horse"]

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
        )

        del text_encoder_one, text_encoder_two

        unet = self.unet()

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

        sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
        # fmt: off
        timesteps = torch.tensor([1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501, 521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761, 781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981], dtype=torch.long, device=self.device)
        # fmt: on

        x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=self.dtype, device=unet.device, generator=torch.Generator(self.device).manual_seed(0))
        x_T_ = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

        out = sdxl_diffusion_loop(
            unet=unet,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            generator=torch.Generator(self.device).manual_seed(0),
            x_T=x_T_,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        del unet

        vae = self.vae()

        out = vae.output_tensor_to_pil(vae.decode(out))[0]
        out.save(f"./test_text_to_image_{self.device}_{self.dtype}.png")
        out = np.array(out).astype(np.int32)

        del vae

        sdxl_pipe = self.sdxl_pipe()

        expected_out = sdxl_pipe(prompt="horse", latents=x_T).images[0]

        del sdxl_pipe

        expected_out = np.array(expected_out).astype(np.int32)

        diff = np.abs(out - expected_out).flatten()
        diff.sort()

        assert scipy.stats.mode(diff).mode < 2

        if self.dtype == torch.float32:
            assert diff.mean() < 2
        elif self.dtype == torch.float16:
            assert diff.mean() < 4
        else:
            assert False

    @torch.no_grad()
    def test_heun(self):
        print("test_heun")

        prompts = ["horse"]

        text_encoder_one = self.text_encoder_one()
        text_encoder_two = self.text_encoder_two()
        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_two.device),
        )

        del text_encoder_one, text_encoder_two

        unet = self.unet()

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

        sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=unet.device)

        out = sdxl_diffusion_loop(
            unet=unet,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            sampler=heun_ode_solver,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        del unet

        vae = self.vae()

        out = vae.output_tensor_to_pil(vae.decode(out))[0]

        del vae

        out.save(f"./test_heun_{self.device}_{self.dtype}.png")

    @torch.no_grad()
    def test_rk4(self):
        print("test_rk4")

        prompts = ["horse"]

        text_encoder_one = self.text_encoder_one()
        text_encoder_two = self.text_encoder_two()
        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_two.device),
        )

        del text_encoder_one, text_encoder_two

        unet = self.unet()

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

        sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=unet.device)

        out = sdxl_diffusion_loop(
            unet=unet,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            sampler=rk4_ode_solver,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        del unet

        vae = self.vae()

        out = vae.output_tensor_to_pil(vae.decode(out))[0]

        del vae

        out.save(f"./test_rk4_{self.device}_{self.dtype}.png")

    @torch.no_grad()
    def test_controlnet(self):
        print("test_controlnet")

        import cv2

        image = Image.open(hf_hub_download("williamberman/misc", "bright_room_with_chair.png", repo_type="dataset")).convert("RGB").resize((1024, 1024))
        image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        image = image[None, :, :, :]

        prompts = ["a beautiful room"]

        text_encoder_one = self.text_encoder_one()
        text_encoder_two = self.text_encoder_two()

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_two.device),
        )

        del text_encoder_one, text_encoder_two

        unet = self.unet()

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

        if self.dtype == torch.float32:
            controlnet = SDXLControlNet.load(hf_hub_download("diffusers/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors"), device=self.device)
        elif self.dtype == torch.float16:
            controlnet = SDXLControlNet.load(hf_hub_download("diffusers/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.fp16.safetensors"), device=self.device)
        else:
            assert False

        sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=unet.device)
        image = image.to(device=controlnet.device, dtype=controlnet.dtype)

        out = sdxl_diffusion_loop(
            unet=unet,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            controlnet=controlnet,
            images=image,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        del controlnet, unet

        vae = self.vae()

        out = vae.output_tensor_to_pil(vae.decode(out))[0]

        del vae

        out.save(f"./test_controlnet_{self.device}_{self.dtype}.png")

    @torch.no_grad()
    def test_adapter(self):
        print("test_adapter")

        import cv2

        image = Image.open(hf_hub_download("williamberman/misc", "bright_room_with_chair.png", repo_type="dataset")).convert("RGB").resize((1024, 1024))
        image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        image = image[None, :, :, :]

        prompts = ["a beautiful room"]

        text_encoder_one = self.text_encoder_one()
        text_encoder_two = self.text_encoder_two()

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_two.device),
        )

        del text_encoder_one, text_encoder_two

        unet = self.unet()

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

        if self.dtype == torch.float32:
            adapter = SDXLAdapter.load(hf_hub_download("TencentARC/t2i-adapter-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors"), device=self.device)
        elif self.dtype == torch.float16:
            adapter = SDXLAdapter.load(hf_hub_download("TencentARC/t2i-adapter-canny-sdxl-1.0", "diffusion_pytorch_model.fp16.safetensors"), device=self.device)
        else:
            assert False

        sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=unet.device)
        image = image.to(device=adapter.device, dtype=adapter.dtype)

        out = sdxl_diffusion_loop(
            unet=unet,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            adapter=adapter,
            images=image,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        del adapter, unet

        vae = self.vae()

        out = vae.output_tensor_to_pil(vae.decode(out))[0]

        del vae

        out.save(f"./test_adapter_{self.device}_{self.dtype}.png")


class TrainControlnetInpaintingTests:
    def __init__(self, device):
        print(f"loading train controlnet inpainting tests {device}")
        x = train.init_train_controlnet(
            namedtuple_helper(
                controlnet_resume_from=None,
                use_8bit_adam=True,
                optimizer_resume_from=None,
                learning_rate=0.00001,
            ),
            make_dataloader=False,
        )

        self.tokenizer_one = x["tokenizer_one"]
        self.tokenizer_two = x["tokenizer_two"]
        self.text_encoder_one = x["text_encoder_one"]
        self.text_encoder_two = x["text_encoder_two"]
        self.vae = x["vae"]
        self.sigmas = x["sigmas"]
        self.unet = x["unet"]
        self.controlnet = x["controlnet"]
        self.optimizer = x["optimizer"]

    def test_log_validation(self):
        print("test_log_validation")

        timesteps = torch.tensor([0], dtype=torch.long, device=self.sigmas.device)

        output_images, conditioning_images = train.log_validation_train_controlnet(
            self.tokenizer_one,
            self.text_encoder_one,
            self.tokenizer_two,
            self.text_encoder_two,
            self.vae,
            self.sigmas.to(self.unet.dtype),
            self.unet,
            [
                "https://huggingface.co/datasets/williamberman/misc/resolve/main/bright_room_with_chair.png",
                "https://huggingface.co/datasets/williamberman/misc/resolve/main/couple_sitting_on_bench_infront_of_lake.png",
            ],
            ["bright room with chair", "couple sitting on bench in front of lake"],
            2,
            get_controlnet_inpainting_conditioning_image,
            controlnet=self.controlnet,
            timesteps=timesteps,
        )

        assert len(output_images) == 4
        assert conditioning_images is not None
        assert len(conditioning_images) == 2


def test_save_checkpoint():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMP_DIR", None)) as tmpdir:
        train.make_save_checkpoint(
            output_dir=tmpdir,
            checkpoints_total_limit=None,
            training_step=0,
        )

        assert set(os.listdir(tmpdir)) == {"checkpoint-0"}


def test_save_checkpoint_checkpoints_total_limit():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMP_DIR", None)) as tmpdir:
        os.mkdir(os.path.join(tmpdir, "checkpoint-0"))
        os.mkdir(os.path.join(tmpdir, "checkpoint-1"))

        train.make_save_checkpoint(
            output_dir=tmpdir,
            checkpoints_total_limit=3,
            training_step=2,
        )

        # removes none
        assert set(os.listdir(tmpdir)) == {"checkpoint-0", "checkpoint-1", "checkpoint-2"}

        train.make_save_checkpoint(
            output_dir=tmpdir,
            checkpoints_total_limit=3,
            training_step=3,
        )

        # removes one
        assert set(os.listdir(tmpdir)) == {"checkpoint-1", "checkpoint-2", "checkpoint-3"}

        os.mkdir(os.path.join(tmpdir, "checkpoint-4"))

        train.make_save_checkpoint(
            output_dir=tmpdir,
            checkpoints_total_limit=3,
            training_step=5,
        )

        # removes two
        assert set(os.listdir(tmpdir)) == {"checkpoint-3", "checkpoint-4", "checkpoint-5"}


def test_controlnet_inpainting_main():
    print("test_controlnet_inpainting_main")

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMP_DIR", None)) as tmpdir:
        training_config = train.TrainingConfig(
            output_dir=tmpdir,
            controlnet="inpainting",
            learning_rate=0.00001,
            gradient_accumulation_steps=1,
            mixed_precision=torch.float16,
            max_train_steps=2,
            dataloader="data.wds_dataloader",
            shuffle_buffer_size=1000,
            proportion_empty_prompts=0.1,
            batch_size=2,
            sdxl_synthetic_dataset=True,
            validation_steps=1,
            num_validation_timesteps=2,
            validation_image_conditioning="data.get_controlnet_inpainting_conditioning_image",
            num_validation_images=1,
            validation_images=[
                "https://huggingface.co/datasets/williamberman/misc/resolve/main/bright_room_with_chair.png",
            ],
            validation_prompts=["bright room with chair"],
            checkpointing_steps=1,
            checkpoints_total_limit=5,
            project_name="nano_diffusion_testing",
            training_run_name="test_controlnet_inpainting",
            train_shards=["pipe:aws s3 cp s3://muse-datasets/sdxl-synthetic-dataset/0/{00000..00011}.tar -"],
            use_8bit_adam=True,
        )

        train.main(training_config)

        assert set(os.listdir(tmpdir)) == {"controlnet.safetensors", "optimizer.bin", "checkpoint-1"}

        training_config_resume_from = dataclasses.asdict(training_config)
        training_config_resume_from.update(
            dict(
                controlnet_resume_from=os.path.join(tmpdir, "checkpoint-1", "controlnet.safetensors"),
                optimizer_resume_from=os.path.join(tmpdir, "checkpoint-1", "optimizer.bin"),
                start_step=2,
                max_train_steps=3,
            )
        )
        training_config_resume_from = train.TrainingConfig(**training_config_resume_from)

        train.main(training_config_resume_from)

        assert set(os.listdir(tmpdir)) == {"controlnet.safetensors", "optimizer.bin", "checkpoint-1", "checkpoint-2"}


def namedtuple_helper(**kwargs):
    return namedtuple("anon", kwargs.keys())(**kwargs)


if __name__ == "__main__":
    torch.cuda.set_device(train.device)
    dist.init_process_group("nccl")

    for dtype in [torch.float16, torch.float32]:
        tests = InferenceTests("cuda", dtype)
        tests.test_sdxl_vae()
        tests.test_sdxl_unet()
        tests.test_text_to_image()
        tests.test_heun()
        tests.test_rk4()
        tests.test_controlnet()
        tests.test_adapter()

    test_save_checkpoint()
    test_save_checkpoint_checkpoints_total_limit()

    tests = TrainControlnetInpaintingTests(0)
    tests.test_log_validation()

    test_controlnet_inpainting_main()

    print("All tests passed!")
