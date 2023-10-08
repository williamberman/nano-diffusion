import dataclasses
import os
import tempfile

import numpy as np
import scipy
import torch
import torch.distributed as dist
from diffusers import (AutoencoderKL, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import train
from data import get_controlnet_inpainting_conditioning_image
from diffusion import (heun_ode_solver, make_sigmas, rk4_ode_solver,
                       sdxl_diffusion_loop, sdxl_text_conditioning,
                       set_with_tqdm)
from models import (SDXLAdapter, SDXLCLIPOne, SDXLCLIPTwo, SDXLControlNet,
                    SDXLUNet, SDXLVae, make_clip_tokenizer_one_from_hub,
                    make_clip_tokenizer_two_from_hub,
                    set_attention_implementation)

set_with_tqdm(True)


class InferenceTests:
    def __init__(self, device, dtype):
        print(f"loading inference tests {device} {dtype}")

        self.tokenizer_one = make_clip_tokenizer_one_from_hub()
        self.tokenizer_two = make_clip_tokenizer_two_from_hub()

        if dtype == torch.float32:
            text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
            text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)
            vae = SDXLVae.load_fp16_fix(device=device)
            unet = SDXLUNet.load_fp32(device=device)

            vae_ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
            vae_.to(device=device)

            unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
            unet_.to(device=device)

            sdxl_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet_, vae=vae_)
            sdxl_pipe.to(device)
            sdxl_pipe.set_progress_bar_config(disable=True)
        elif dtype == torch.float16:
            text_encoder_one = SDXLCLIPOne.load_fp16(device=device)
            text_encoder_two = SDXLCLIPTwo.load_fp16(device=device)
            vae = SDXLVae.load_fp16_fix(device=device)
            vae.to(dtype=torch.float16)
            unet = SDXLUNet.load_fp16(device=device)

            vae_ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            vae_.to(device=device)

            unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16, variant="fp16")
            unet_.to(device=device)

            sdxl_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16, unet=unet_, vae=vae_)
            sdxl_pipe.to(device)
            sdxl_pipe.set_progress_bar_config(disable=True)
        else:
            assert False

        self.device = device
        self.dtype = dtype

        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.vae_ = vae_
        self.unet = unet
        self.unet_ = unet_
        self.sdxl_pipe = sdxl_pipe

    @torch.no_grad()
    def test_sdxl_vae(self):
        print("test_sdxl_vae")

        image = Image.open(hf_hub_download("williamberman/misc", "two_birds_on_branch.png", repo_type="dataset"))
        image = image.convert("RGB")
        image = image.resize((1024, 1024))
        image = self.vae.input_pil_to_tensor(image)
        image = image.to(dtype=self.vae.dtype, device=self.vae.device)

        encoder_output = self.vae.encode(image, generator=torch.Generator(self.device).manual_seed(0))

        expected_encoder_output = self.vae_.encode(image).latent_dist.sample(generator=torch.Generator(self.device).manual_seed(0))
        expected_encoder_output = expected_encoder_output * self.vae_.config.scaling_factor

        total_diff = (expected_encoder_output.float() - encoder_output.float()).abs().sum()
        assert total_diff == 0

        decoder_output = self.vae.decode(encoder_output)
        expected_decoder_output = self.vae_.decode(expected_encoder_output / self.vae_.config.scaling_factor).sample

        total_diff = (expected_decoder_output.float() - decoder_output.float()).abs().sum()
        assert total_diff == 0

    @torch.no_grad()
    def test_sdxl_unet(self):
        print("test_sdxl_unet")

        prompts = ["horse"]

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            self.text_encoder_one,
            self.text_encoder_two,
            torch.tensor([x.ids for x in self.tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=self.text_encoder_one.device),
            torch.tensor([x.ids for x in self.tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=self.text_encoder_one.device),
        )
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.unet.dtype, device=self.unet.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=self.unet.dtype, device=self.unet.device)

        micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=self.unet.device)

        x_t = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=self.unet.dtype, device=self.unet.device, generator=torch.Generator(self.device).manual_seed(0))

        t = torch.tensor([500], dtype=torch.long, device=self.device)

        unet_output = self.unet(
            x_t=x_t,
            t=t,
            encoder_hidden_states=encoder_hidden_states,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            micro_conditioning=micro_conditioning,
        )

        expected_unet_output = self.unet_(
            sample=x_t, timestep=t, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs={"time_ids": micro_conditioning, "text_embeds": pooled_encoder_hidden_states}
        ).sample

        total_diff = (expected_unet_output.float() - unet_output.float()).abs().sum()

        assert total_diff == 0

    @torch.no_grad()
    def test_text_to_image(self):
        print("test_text_to_image")

        sigmas = make_sigmas(device=self.unet.device).to(dtype=self.unet.dtype)
        # fmt: off
        timesteps = torch.tensor([1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501, 521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761, 781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981], dtype=torch.long, device=self.device)
        # fmt: on

        x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=self.dtype, device=self.unet_.device, generator=torch.Generator(self.device).manual_seed(0))
        x_T_ = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

        out = sdxl_diffusion_loop(
            ["horse"],
            unet=self.unet,
            tokenizer_one=self.tokenizer_one,
            text_encoder_one=self.text_encoder_one,
            tokenizer_two=self.tokenizer_two,
            text_encoder_two=self.text_encoder_two,
            generator=torch.Generator(self.device).manual_seed(0),
            x_T=x_T_,
            timesteps=timesteps,
            sigmas=sigmas,
        )
        out = self.vae.output_tensor_to_pil(self.vae.decode(out))[0]
        out.save(f"./test_text_to_image_{self.device}_{self.dtype}.png")
        out = np.array(out).astype(np.int32)

        expected_out = self.sdxl_pipe(prompt="horse", latents=x_T).images[0]
        expected_out = np.array(expected_out).astype(np.int32)

        diff = np.abs(out - expected_out).flatten()
        diff.sort()

        assert scipy.stats.mode(diff).mode < 2

        if self.dtype == torch.float32:
            assert diff.mean() < 1
        elif self.dtype == torch.float16:
            assert diff.mean() < 3
        else:
            assert False

    @torch.no_grad()
    def test_heun(self):
        print("test_heun")

        sigmas = make_sigmas(device=self.unet.device).to(dtype=self.unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=self.unet.device)

        out = sdxl_diffusion_loop(
            ["horse"],
            unet=self.unet,
            tokenizer_one=self.tokenizer_one,
            text_encoder_one=self.text_encoder_one,
            tokenizer_two=self.tokenizer_two,
            text_encoder_two=self.text_encoder_two,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            sampler=heun_ode_solver,
        )
        out = self.vae.output_tensor_to_pil(self.vae.decode(out))[0]
        out.save(f"./test_heun_{self.device}_{self.dtype}.png")

    @torch.no_grad()
    def test_rk4(self):
        print("test_rk4")

        sigmas = make_sigmas(device=self.unet.device).to(dtype=self.unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=self.unet.device)

        out = sdxl_diffusion_loop(
            ["horse"],
            unet=self.unet,
            tokenizer_one=self.tokenizer_one,
            text_encoder_one=self.text_encoder_one,
            tokenizer_two=self.tokenizer_two,
            text_encoder_two=self.text_encoder_two,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            sampler=rk4_ode_solver,
        )
        out = self.vae.output_tensor_to_pil(self.vae.decode(out))[0]
        out.save(f"./test_rk4_{self.device}_{self.dtype}.png")

    @torch.no_grad()
    def test_controlnet(self):
        print("test_controlnet")

        if self.dtype == torch.float32:
            controlnet = SDXLControlNet.load(hf_hub_download("diffusers/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors"), device=self.device)
        elif self.dtype == torch.float16:
            controlnet = SDXLControlNet.load(hf_hub_download("diffusers/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.fp16.safetensors"), device=self.device)
        else:
            assert False

        import cv2

        sigmas = make_sigmas(device=self.unet.device).to(dtype=self.unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=self.unet.device)

        image = Image.open(hf_hub_download("williamberman/misc", "bright_room_with_chair.png", repo_type="dataset")).convert("RGB").resize((1024, 1024))
        image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        image = image[None, :, :, :].to(device=self.device, dtype=controlnet.dtype)

        out = sdxl_diffusion_loop(
            ["a beautiful room"],
            unet=self.unet,
            tokenizer_one=self.tokenizer_one,
            text_encoder_one=self.text_encoder_one,
            tokenizer_two=self.tokenizer_two,
            text_encoder_two=self.text_encoder_two,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            controlnet=controlnet,
            images=image,
        )
        out = self.vae.output_tensor_to_pil(self.vae.decode(out))[0]
        out.save(f"./test_controlnet_{self.device}_{self.dtype}.png")

    def test_adapter(self):
        print("test_adapter")

        if self.dtype == torch.float32:
            adapter = SDXLAdapter.load(hf_hub_download("TencentARC/t2i-adapter-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors"), device=self.device)
        elif self.dtype == torch.float16:
            adapter = SDXLAdapter.load(hf_hub_download("TencentARC/t2i-adapter-canny-sdxl-1.0", "diffusion_pytorch_model.fp16.safetensors"), device=self.device)
        else:
            assert False

        import cv2

        sigmas = make_sigmas(device=self.unet.device).to(dtype=self.unet.dtype)
        timesteps = torch.linspace(0, sigmas.numel() - 1, 10, dtype=torch.long, device=self.unet.device)

        image = Image.open(hf_hub_download("williamberman/misc", "bright_room_with_chair.png", repo_type="dataset")).convert("RGB").resize((1024, 1024))
        image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        image = image[None, :, :, :].to(device=self.device, dtype=adapter.dtype)

        out = sdxl_diffusion_loop(
            ["a beautiful room"],
            unet=self.unet,
            tokenizer_one=self.tokenizer_one,
            text_encoder_one=self.text_encoder_one,
            tokenizer_two=self.tokenizer_two,
            text_encoder_two=self.text_encoder_two,
            generator=torch.Generator(self.device).manual_seed(0),
            timesteps=timesteps,
            sigmas=sigmas,
            adapter=adapter,
            images=image,
        )
        out = self.vae.output_tensor_to_pil(self.vae.decode(out))[0]
        out.save(f"./test_adapter_{self.device}_{self.dtype}.png")


class TrainControlnetInpaintingTests:
    def __init__(self, device):
        print(f"loading train controlnet inpainting tests {device}")

        tokenizer_one = make_clip_tokenizer_one_from_hub()
        tokenizer_two = make_clip_tokenizer_two_from_hub()

        text_encoder_one = SDXLCLIPOne.load_fp16(device=device)
        text_encoder_one.requires_grad_(False)
        text_encoder_one.eval()

        text_encoder_two = SDXLCLIPTwo.load_fp16(device=device)
        text_encoder_two.requires_grad_(False)
        text_encoder_two.eval()

        vae = SDXLVae.load_fp16_fix(device=device)
        vae.to(torch.float16)
        vae.requires_grad_(False)
        vae.eval()

        sigmas = make_sigmas(device=device)

        unet = SDXLUNet.load_fp16()
        unet.to(device)
        unet.requires_grad_(False)
        unet.eval()

        controlnet = SDXLControlNet.from_unet(unet)
        controlnet.to(device)
        controlnet.train()
        controlnet.requires_grad_(True)
        controlnet = DDP(controlnet, device_ids=[device])

        optimizer = AdamW(controlnet.parameters())

        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.sigmas = sigmas
        self.unet = unet
        self.controlnet = controlnet
        self.optimizer = optimizer

    def test_log_validation_controlnet_inpainting(self):
        print("test_log_validation_controlnet_inpainting")

        timesteps = torch.tensor([0], dtype=torch.long, device=self.sigmas.device)

        output_images, conditioning_images = train.log_validation(
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

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMP_DIR", None)) as tmpdir:
            train.save_checkpoint(
                unet=self.unet,
                output_dir=tmpdir,
                checkpoints_total_limit=None,
                training_step=0,
                optimizer=self.optimizer,
                controlnet=self.controlnet,
            )

            assert set(os.listdir(tmpdir)) == {"checkpoint-0"}
            assert set(os.listdir(os.path.join(tmpdir, "checkpoint-0"))) == {"optimizer.bin", "controlnet.safetensors"}

    def test_save_checkpoint_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMP_DIR", None)) as tmpdir:
            os.mkdir(os.path.join(tmpdir, "checkpoint-0"))
            os.mkdir(os.path.join(tmpdir, "checkpoint-1"))

            train.save_checkpoint(
                unet=self.unet,
                output_dir=tmpdir,
                checkpoints_total_limit=3,
                training_step=2,
                optimizer=self.optimizer,
                controlnet=self.controlnet,
            )

            # removes none
            assert set(os.listdir(tmpdir)) == {"checkpoint-0", "checkpoint-1", "checkpoint-2"}

            train.save_checkpoint(
                unet=self.unet,
                output_dir=tmpdir,
                checkpoints_total_limit=3,
                training_step=3,
                optimizer=self.optimizer,
                controlnet=self.controlnet,
            )

            # removes one
            assert set(os.listdir(tmpdir)) == {"checkpoint-1", "checkpoint-2", "checkpoint-3"}

            os.mkdir(os.path.join(tmpdir, "checkpoint-4"))

            train.save_checkpoint(
                unet=self.unet,
                output_dir=tmpdir,
                checkpoints_total_limit=3,
                training_step=5,
                optimizer=self.optimizer,
                controlnet=self.controlnet,
            )

            # removes two
            assert set(os.listdir(tmpdir)) == {"checkpoint-3", "checkpoint-4", "checkpoint-5"}

    def test(self):
        self.test_log_validation_controlnet_inpainting()
        self.test_save_checkpoint()
        self.test_save_checkpoint_checkpoints_total_limit()


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


if __name__ == "__main__":
    torch.cuda.set_device(train.device)
    dist.init_process_group("nccl")

    set_attention_implementation("torch_2.0_scaled_dot_product")
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        for dtype in [torch.float16, torch.float32]:
            tests = InferenceTests("cuda", dtype)
            tests.test_sdxl_vae()
            tests.test_sdxl_unet()
            tests.test_text_to_image()
            tests.test_heun()
            tests.test_rk4()
            tests.test_controlnet()
            tests.test_adapter()

    TrainControlnetInpaintingTests(0).test()
    test_controlnet_inpainting_main()
    print("All tests passed!")
