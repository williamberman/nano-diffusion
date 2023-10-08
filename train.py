import argparse
import importlib
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from diffusion import make_sigmas, sdxl_diffusion_loop
from models import (SDXLAdapter, SDXLCLIPOne, SDXLCLIPTwo, SDXLControlNet,
                    SDXLUNet, SDXLVae, make_clip_tokenizer_one_from_hub,
                    make_clip_tokenizer_two_from_hub, sdxl_text_conditioning)

try:
    import safetensors.torch

    has_safetensors = True
except ImportError:
    has_safetensors = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

device = int(os.environ["LOCAL_RANK"])

# TODO - test this and double check the temporarily disabling gradient syncronization during forward pass of
# gradient accumulation


@dataclass
class TrainingConfig:
    output_dir: str
    train_shards: str
    dataloader: str

    # additional networks
    adapter: Optional[Literal["openpose"]] = None
    controlnet: Optional[Literal["canny", "inpainting"]] = None

    # training
    learning_rate: float = 0.00001
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[torch.dtype] = None
    max_train_steps: int = 30_000
    use_8bit_adam: bool = False

    # data
    shuffle_buffer_size: int = 1000
    proportion_empty_prompts: float = 0.1
    batch_size: int = 8
    sdxl_synthetic_dataset: bool = False
    validation_image_conditioning: Optional[str] = None

    # validation
    validation_steps: int = 500
    num_validation_images: int = 2
    num_validation_timesteps: Optional[int] = None
    validation_prompts: Optional[List[str]] = None
    validation_images: Optional[List[str]] = None

    # checkpointing
    checkpointing_steps: int = 1000
    checkpoints_total_limit: int = 5
    unet_resume_from: Optional[str] = None
    controlnet_resume_from: Optional[str] = None
    adapter_resume_from: Optional[str] = None
    optimizer_resume_from: Optional[str] = None
    start_step: int = 0

    # wandb
    project_name: Optional[str] = None
    training_run_name: Optional[str] = None
    log_to_wandb: bool = True


def main(training_config):
    if dist.get_rank() == 0:
        os.makedirs(training_config.output_dir, exist_ok=True)

        if training_config.log_to_wandb:
            wandb.init(
                name=training_config.training_run_name,
                project=training_config.project_name,
                config=training_config,
            )

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

    parameters = []

    if training_config.controlnet is None and training_config.adapter is None:
        if training_config.unet_resume_from is None:
            unet = SDXLUNet.load_fp32()
        else:
            unet = SDXLUNet.load(training_config.unet_resume_from, device=device)
        unet.requires_grad_(True)
        unet.train()
        unet = DDP(unet, device_ids=[device])

        parameters.extend(unet.module.parameters())
    else:
        unet = SDXLUNet.load_fp16(device=device)
        unet.requires_grad_(False)
        unet.eval()

    if training_config.controlnet is not None:
        if training_config.controlnet_resume_from is None:
            controlnet = SDXLControlNet.from_unet(unet)
            controlnet.to(device)
        else:
            controlnet = SDXLControlNet.load(training_config.controlnet_resume_from, device=device)
        controlnet.train()
        controlnet.requires_grad_(True)
        controlnet = DDP(controlnet, device_ids=[device])

        parameters.extend(controlnet.module.parameters())
    else:
        controlnet = None

    if training_config.adapter is not None:
        if training_config.adapter_resume_from is None:
            adapter = SDXLAdapter()
            adapter.to(device)
        else:
            adapter = SDXLAdapter.load(training_config.adapter_resume_from, device=device)
        adapter.train()
        adapter.requires_grad_(True)
        adapter = DDP(adapter, device_ids=[device])

        parameters.extend(adapter.module.parameters())
    else:
        adapter = None

    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer = bnb.optim.AdamW8bit(parameters, lr=training_config.learning_rate)
    else:
        optimizer = AdamW(parameters, lr=training_config.learning_rate)

    # TODO - make LR configurable for more than finetuning
    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    if training_config.optimizer_resume_from is not None:
        optimizer.load_state_dict(torch.load(training_config.optimizer_resume_from, map_location=torch.device(device)))

    module, fn = training_config.dataloader.split(".")
    dataloader_fn = getattr(importlib.import_module(module), fn)
    dataloader = iter(dataloader_fn(training_config, tokenizer_one, tokenizer_two))

    scaler = GradScaler(enabled=training_config.mixed_precision == torch.float16)

    for training_step in range(training_config.start_step, training_config.max_train_steps):
        accumulated_loss = None

        for _ in range(training_config.gradient_accumulation_steps):
            batch = next(dataloader)

            loss = train_step(
                text_encoder_one=text_encoder_one,
                text_encoder_two=text_encoder_two,
                vae=vae,
                unet=unet,
                mixed_precision=training_config.mixed_precision,
                batch=batch,
                controlnet=controlnet,
                adapter=adapter,
                sigmas=sigmas,
            )

            loss = loss / training_config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if accumulated_loss is None:
                accumulated_loss = loss.detach()
            else:
                accumulated_loss += loss.detach()

        scaler.unscale_(optimizer)

        clip_grad_norm_(parameters, 1.0)

        scaler.step(optimizer)

        lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        scaler.update()

        if training_step != 0 and training_step % training_config.checkpointing_steps == 0:
            if dist.get_rank() == 0:
                save_checkpoint(
                    unet=unet,
                    output_dir=training_config.output_dir,
                    checkpoints_total_limit=training_config.checkpoints_total_limit,
                    training_step=training_step,
                    optimizer=optimizer,
                    controlnet=controlnet,
                    adapter=adapter,
                )

            dist.barrier()

        if training_config.log_to_wandb and dist.get_rank() == 0 and training_step != 0 and training_step % training_config.validation_steps == 0:
            logger.info("Running validation")

            module, fn = training_config.validation_image_conditioning.split(".")
            validation_image_conditioning = getattr(importlib.import_module(module), fn)

            if training_config.num_validation_timesteps is not None:
                validation_timesteps = torch.linspace(0, sigmas.numel() - 1, training_config.num_validation_timesteps, dtype=torch.long, device=unet.device)
            else:
                validation_timesteps = None

            output_images, conditioning_images = log_validation(
                tokenizer_one=tokenizer_one,
                text_encoder_one=text_encoder_one,
                tokenizer_two=tokenizer_two,
                text_encoder_two=text_encoder_two,
                vae=vae,
                sigmas=sigmas.to(unet.module.dtype if isinstance(unet, DDP) else unet.dtype),
                unet=unet,
                num_validation_images=training_config.num_validation_images,
                validation_prompts=training_config.validation_prompts,
                validation_images=training_config.validation_images,
                validation_image_conditioning=validation_image_conditioning,
                adapter=adapter,
                controlnet=controlnet,
                timesteps=validation_timesteps,
            )

            wandb.log({"validation": output_images}, step=training_step)

            if conditioning_images is not None:
                wandb.log({"validation_conditioning": conditioning_images}, step=training_step)

        if dist.get_rank() == 0:
            loss = accumulated_loss.item()
            lr = lr_scheduler.get_last_lr()[0]
            logger.info(f"Step {training_step}: loss={loss}, lr={lr}")
            if training_config.log_to_wandb:
                wandb.log({"loss": loss, "lr": lr}, step=training_step)

    dist.barrier()

    if dist.get_rank() == 0:
        save_models(unet, training_config.output_dir, optimizer=optimizer, controlnet=controlnet, adapter=adapter)


def train_step(text_encoder_one, text_encoder_two, vae, sigmas, unet, batch, controlnet=None, adapter=None, mixed_precision=None):
    with torch.no_grad():
        if isinstance(unet, DDP):
            unet_dtype = unet.module.dtype
            unet_device = unet.module.device
        else:
            unet_dtype = unet.dtype
            unet_device = unet.device

        micro_conditioning = batch["micro_conditioning"].to(device=unet_device)

        image = batch["image"].to(vae.device, dtype=vae.dtype)
        latents = vae.encode(image).to(dtype=unet_dtype)

        text_input_ids_one = batch["text_input_ids_one"].to(text_encoder_one.device)
        text_input_ids_two = batch["text_input_ids_two"].to(text_encoder_two.device)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet_dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, sigmas.numel(), (bsz,), device=unet_device)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)[:, None, None, None]

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

        if "conditioning_image" in batch:
            conditioning_image = batch["conditioning_image"].to(unet_device)
        else:
            conditioning_image = None

    with torch.autocast(
        "cuda",
        mixed_precision,
        enabled=mixed_precision is not None,
    ):
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        add_to_down_block_outputs = None

        if adapter is not None:
            add_to_down_block_outputs = adapter(conditioning_image)

        if controlnet is not None:
            controlnet_out = controlnet(
                x_t=scaled_noisy_latents,
                t=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                micro_conditioning=micro_conditioning,
                pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                controlnet_cond=conditioning_image,
            )

            down_block_additional_residuals = controlnet_out["down_block_res_samples"]
            mid_block_additional_residual = controlnet_out["mid_block_res_sample"]

        model_pred = unet(
            x_t=scaled_noisy_latents,
            t=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            add_to_down_block_outputs=add_to_down_block_outputs,
        )

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


@torch.no_grad()
def log_validation(
    tokenizer_one,
    text_encoder_one,
    tokenizer_two,
    text_encoder_two,
    vae,
    sigmas,
    unet,
    validation_images,
    validation_prompts,
    num_validation_images,
    validation_image_conditioning,
    adapter=None,
    controlnet=None,
    timesteps=None,
):
    if isinstance(unet, DDP):
        unet_ = unet.module
        unet_.eval()
        unet_set_to_eval = True
    else:
        unet_ = unet
        unet_set_to_eval = False

    if adapter is not None:
        adapter_ = adapter.module
        adapter_.eval()
    else:
        adapter_ = None

    if controlnet is not None:
        controlnet_ = controlnet.module
        controlnet_.eval()
    else:
        controlnet_ = None

    conditioning_images = None
    formatted_validation_images = None

    if validation_images is not None:
        formatted_validation_images = []
        conditioning_images = []

        for validation_image_path in validation_images:
            validation_image_path: str = validation_image_path
            if validation_image_path.startswith(("https://", "http://")):
                import requests

                validation_image = Image.open(requests.get(validation_image_path, stream=True).raw)
            else:
                validation_image = Image.open(validation_image_path)

            validation_image = validation_image.convert("RGB")
            validation_image = validation_image.resize((1024, 1024))

            conditioning_image = validation_image_conditioning(validation_image)
            formatted_validation_images.append(conditioning_image["conditioning_image"][None, :, :, :])
            conditioning_images.append(wandb.Image(conditioning_image["conditioning_image_as_pil"]))

    generator = torch.Generator(unet_.device).manual_seed(0)

    output_images = []

    for formatted_validation_image, validation_prompt in zip(formatted_validation_images, validation_prompts):
        for _ in range(num_validation_images):
            x_0 = sdxl_diffusion_loop(
                prompts=validation_prompt,
                images=formatted_validation_image,
                unet=unet_,
                tokenizer_one=tokenizer_one,
                text_encoder_one=text_encoder_one,
                tokenizer_two=tokenizer_two,
                text_encoder_two=text_encoder_two,
                controlnet=controlnet_,
                adapter=adapter_,
                sigmas=sigmas,
                timesteps=timesteps,
                generator=generator,
            )

            x_0 = vae.decode(x_0.to(vae.dtype))
            x_0 = vae.output_tensor_to_pil(x_0)[0]

            output_images.append(wandb.Image(x_0, caption=validation_prompt))

    if unet_set_to_eval:
        unet_.train()

    if adapter_ is not None:
        adapter_.train()

    if controlnet_ is not None:
        controlnet_.train()

    return output_images, conditioning_images


def save_checkpoint(unet, output_dir, checkpoints_total_limit, training_step, optimizer, controlnet=None, adapter=None):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f"checkpoint-{training_step}")

    os.makedirs(save_path, exist_ok=True)

    save_models(unet, save_path, optimizer=optimizer, controlnet=controlnet, adapter=adapter)


def save_models(unet, save_path: str, optimizer, controlnet=None, adapter=None):
    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    except RuntimeError as err:
        # TODO - RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 2075490688 vs 2075490580
        logger.warning(f"failed to save optimizer {err}")

    if has_safetensors:
        if isinstance(unet, DDP):
            safetensors.torch.save_file(unet.module.state_dict(), os.path.join(save_path, "unet.safetensors"))

        if controlnet is not None:
            safetensors.torch.save_file(controlnet.module.state_dict(), os.path.join(save_path, "controlnet.safetensors"))

        if adapter is not None:
            safetensors.torch.save_file(adapter.module.state_dict(), os.path.join(save_path, "adapter.safetensors"))
    else:
        if isinstance(unet, DDP):
            torch.save(unet.module.state_dict(), os.path.join(save_path, "unet.bin"))

        if controlnet is not None:
            torch.save(controlnet.module.state_dict(), os.path.join(save_path, "controlnet.bin"))

        if adapter is not None:
            torch.save(adapter.module.state_dict(), os.path.join(save_path, "adapter.bin"))

    logger.info(f"Saved to {save_path}")


def load_config():
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, required=False)
    args = args.parse_args()

    if args.config_path is None and "NANO_DIFFUSION_TRAINING_CONFIG" not in os.environ:
        raise ValueError(
            f"Must set either the environment variable `'NANO_DIFFUSION_TRAINING_CONFIG'` or pass the `--config_path` argument to point to a yaml file containing config for the training run."
        )

    if args.config_path is not None and "NANO_DIFFUSION_TRAINING_CONFIG" in os.environ:
        raise ValueError(
            f"Must set only one of the environment variable `'NANO_DIFFUSION_TRAINING_CONFIG'` or pass the `--config_path` argument to point to a yaml file containing config for the training run."
        )

    if args.config_path is not None:
        config_path = args.config_path
    elif "NANO_DIFFUSION_TRAINING_CONFIG" in os.environ:
        config_path = os.environ["NANO_DIFFUSION_TRAINING_CONFIG"]
    else:
        assert False

    with open(config_path, "r") as f:
        yaml_config: Dict = yaml.safe_load(f.read())

    if "mixed_precision" not in yaml_config or yaml_config["mixed_precision"] is None or yaml_config["mixed_precision"] == "no":
        yaml_config["mixed_precision"] = None
    elif yaml_config["mixed_precision"] == "fp16":
        yaml_config["mixed_precision"] = torch.float16
    elif yaml_config["mixed_precision"] == "bf16":
        yaml_config["mixed_precision"] = torch.bfloat16
    else:
        assert False

    training_config = TrainingConfig(**yaml_config)

    return training_config


if __name__ == "__main__":
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    main(load_config())
