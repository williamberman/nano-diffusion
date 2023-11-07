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
from tokenizers import Tokenizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from diffusion import (
    euler_ode_solver, make_sigmas, sdxl_diffusion_loop, sdxl_eps_theta_,
    sdxl_eps_theta_unet_inpainting,
    sdxl_eps_theta_unet_inpainting_cross_attention_conditioning, set_with_tqdm)
from ema_model import EMAModel
from models import (SDXLCLIPOne, SDXLCLIPTwo, SDXLControlNet, SDXLUNet,
                    SDXLUNetInpainting,
                    SDXLUNetInpaintingCrossAttentionConditioning, SDXLVae,
                    make_clip_tokenizer_one_from_hub,
                    make_clip_tokenizer_two_from_hub, sdxl_text_conditioning,
                    set_attention_checkpoint_kv)

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

# TODO - double check the temporarily disabling gradient syncronization during forward pass of
# gradient accumulation


@dataclass
class TrainingConfig:
    output_dir: str
    train_shards: str
    dataloader: Optional[str] = None

    # TODO - adapter and controlnet should just be rolled into train_type
    # additional networks
    adapter: Optional[Literal["openpose"]] = None
    controlnet: Optional[Literal["canny", "inpainting"]] = None
    train_type: Optional[Literal["ema_unet_inpainting", "unet_inpainting", "unet_text_to_image", "unet_inpainting_cross_attention_conditioning"]] = None
    gradient_checkpointing: bool = False

    # training
    learning_rate: float = 0.00001
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[torch.dtype] = None
    max_train_steps: int = 30_000
    use_8bit_adam: bool = False
    ema_decay: Optional[float] = None
    ema_power: Optional[float] = None

    # data
    shuffle_buffer_size: int = 1000
    proportion_empty_prompts: float = 0.1
    batch_size: int = 8
    sdxl_synthetic_dataset: bool = False
    validation_image_conditioning: Optional[str] = None

    # validation
    validation_steps: int = 500
    num_validation_images: int = 2
    num_validation_timesteps: int = 50
    validation_prompts: Optional[List[str]] = None
    validation_images: Optional[List[str]] = None
    with_tqdm: bool = False

    # checkpointing
    checkpointing_steps: int = 1000
    checkpoints_total_limit: int = 5
    unet_resume_from: Optional[str] = None
    ema_unet_resume_from: Optional[str] = None
    controlnet_resume_from: Optional[str] = None
    adapter_resume_from: Optional[str] = None
    optimizer_resume_from: Optional[str] = None
    start_step: int = 0

    # wandb
    project_name: Optional[str] = None
    training_run_name: Optional[str] = None
    log_to_wandb: bool = True


def main(training_config: TrainingConfig):
    if dist.get_rank() == 0:
        os.makedirs(training_config.output_dir, exist_ok=True)

        if training_config.log_to_wandb:
            wandb.init(
                name=training_config.training_run_name,
                project=training_config.project_name,
                config=training_config,
            )

    set_with_tqdm(training_config.with_tqdm)

    if training_config.controlnet is not None:
        x = init_train_controlnet(training_config)
        tokenizer_one = x["tokenizer_one"]
        tokenizer_two = x["tokenizer_two"]
        text_encoder_one = x["text_encoder_one"]
        text_encoder_two = x["text_encoder_two"]
        vae = x["vae"]
        sigmas = x["sigmas"]
        unet = x["unet"]
        controlnet = x["controlnet"]
        optimizer = x["optimizer"]
        lr_scheduler = x["lr_scheduler"]
        dataloader = x["dataloader"]
        parameters = x["parameters"]
    elif training_config.train_type == "ema_unet_inpainting":
        x = init_train_ema_unet_inpainting(training_config)
        tokenizer_one = x["tokenizer_one"]
        tokenizer_two = x["tokenizer_two"]
        text_encoder_one = x["text_encoder_one"]
        text_encoder_two = x["text_encoder_two"]
        vae = x["vae"]
        sigmas = x["sigmas"]
        unet = x["unet"]
        ema_unet = x["ema_unet"]
        optimizer = x["optimizer"]
        lr_scheduler = x["lr_scheduler"]
        dataloader = x["dataloader"]
        parameters = x["parameters"]
    elif training_config.train_type == "unet_inpainting":
        x = init_train_unet_inpainting(training_config)
        tokenizer_one = x["tokenizer_one"]
        tokenizer_two = x["tokenizer_two"]
        text_encoder_one = x["text_encoder_one"]
        text_encoder_two = x["text_encoder_two"]
        vae = x["vae"]
        sigmas = x["sigmas"]
        unet = x["unet"]
        optimizer = x["optimizer"]
        lr_scheduler = x["lr_scheduler"]
        dataloader = x["dataloader"]
        parameters = x["parameters"]
    elif training_config.train_type == "unet_text_to_image":
        x = init_train_unet_text_to_image(training_config)
        tokenizer_one = x["tokenizer_one"]
        tokenizer_two = x["tokenizer_two"]
        text_encoder_one = x["text_encoder_one"]
        text_encoder_two = x["text_encoder_two"]
        vae = x["vae"]
        sigmas = x["sigmas"]
        unet = x["unet"]
        optimizer = x["optimizer"]
        lr_scheduler = x["lr_scheduler"]
        dataloader = x["dataloader"]
        parameters = x["parameters"]
    elif training_config.train_type == "unet_inpainting_cross_attention_conditioning":
        x = init_train_unet_inpainting_cross_attention_conditioning(training_config)
        tokenizer_one = x["tokenizer_one"]
        tokenizer_two = x["tokenizer_two"]
        text_encoder_one = x["text_encoder_one"]
        text_encoder_two = x["text_encoder_two"]
        vae = x["vae"]
        sigmas = x["sigmas"]
        unet = x["unet"]
        optimizer = x["optimizer"]
        lr_scheduler = x["lr_scheduler"]
        dataloader = x["dataloader"]
        parameters = x["parameters"]
    else:
        assert False

    dataloader = iter(dataloader)

    scaler = GradScaler(enabled=training_config.mixed_precision == torch.float16)

    for training_step in range(training_config.start_step, training_config.max_train_steps):
        accumulated_loss = None

        for _ in range(training_config.gradient_accumulation_steps):
            batch = next(dataloader)

            if training_config.controlnet is not None:
                loss = train_step_train_controlnet(
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    unet=unet,
                    batch=batch,
                    controlnet=controlnet,
                    sigmas=sigmas,
                    training_config=training_config,
                )
            elif training_config.train_type == "ema_unet_inpainting":
                loss = train_step_train_ema_unet_inpainting(
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    unet=unet,
                    batch=batch,
                    sigmas=sigmas,
                    training_config=training_config,
                )
            elif training_config.train_type == "unet_inpainting":
                loss = train_step_train_unet_inpainting(
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    unet=unet,
                    batch=batch,
                    sigmas=sigmas,
                    training_config=training_config,
                )
            elif training_config.train_type == "unet_text_to_image":
                loss = train_step_train_unet_text_to_image(
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    unet=unet,
                    batch=batch,
                    sigmas=sigmas,
                    training_config=training_config,
                )
            elif training_config.train_type == "unet_inpainting_cross_attention_conditioning":
                loss = train_step_train_unet_inpainting_cross_attention_conditioning(
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    unet=unet,
                    batch=batch,
                    sigmas=sigmas,
                    training_config=training_config,
                )
            else:
                assert False

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

        if training_config.train_type == "ema_unet_inpainting":
            ema_unet.step(unet.parameters())

        if training_step != 0 and training_step % training_config.checkpointing_steps == 0:
            if dist.get_rank() == 0:
                save_path = make_save_checkpoint(
                    output_dir=training_config.output_dir,
                    checkpoints_total_limit=training_config.checkpoints_total_limit,
                    training_step=training_step,
                )

                if training_config.controlnet is not None:
                    save_models_train_controlnet(save_path, optimizer=optimizer, controlnet=controlnet)
                elif training_config.train_type == "ema_unet_inpainting":
                    save_models_train_ema_unet_inpainting(save_path, optimizer=optimizer, unet=unet, ema_unet=ema_unet)
                elif training_config.train_type == "unet_inpainting":
                    save_models_train_unet_inpainting(save_path, optimizer=optimizer, unet=unet)
                elif training_config.train_type == "unet_text_to_image":
                    save_models_train_unet_text_to_image(save_path, optimizer=optimizer, unet=unet)
                elif training_config.train_type == "unet_inpainting_cross_attention_conditioning":
                    save_models_train_unet_inpainting_cross_attention_conditioning(save_path, optimizer=optimizer, unet=unet)
                else:
                    assert False

            dist.barrier()

        if training_config.log_to_wandb and dist.get_rank() == 0 and training_step != 0 and training_step % training_config.validation_steps == 0:
            logger.info("Running validation")

            validation_timesteps = torch.linspace(0, sigmas.numel() - 1, training_config.num_validation_timesteps, dtype=torch.long, device=unet.device)

            if training_config.controlnet is not None:
                # TODO - get rid of this
                module, fn = training_config.validation_image_conditioning.split(".")
                validation_image_conditioning = getattr(importlib.import_module(module), fn)

                log_validation_train_controlnet(
                    tokenizer_one=tokenizer_one,
                    text_encoder_one=text_encoder_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    sigmas=sigmas.to(unet.dtype),
                    unet=unet,
                    num_validation_images=training_config.num_validation_images,
                    validation_prompts=training_config.validation_prompts,
                    validation_images=training_config.validation_images,
                    validation_image_conditioning=validation_image_conditioning,
                    controlnet=controlnet,
                    timesteps=validation_timesteps,
                    training_step=training_step,
                )
            elif training_config.train_type == "ema_unet_inpainting":
                log_validation_train_ema_unet_inpainting(
                    tokenizer_one=tokenizer_one,
                    text_encoder_one=text_encoder_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    sigmas=sigmas.to(unet.module.dtype),
                    unet=unet,
                    ema_unet=ema_unet,
                    num_validation_images=training_config.num_validation_images,
                    validation_prompts=training_config.validation_prompts,
                    validation_images=training_config.validation_images,
                    timesteps=validation_timesteps,
                    training_step=training_step,
                )
            elif training_config.train_type == "unet_inpainting":
                log_validation_train_unet_inpainting(
                    tokenizer_one=tokenizer_one,
                    text_encoder_one=text_encoder_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    sigmas=sigmas.to(unet.module.dtype),
                    unet=unet,
                    num_validation_images=training_config.num_validation_images,
                    validation_prompts=training_config.validation_prompts,
                    validation_images=training_config.validation_images,
                    timesteps=validation_timesteps,
                    training_step=training_step,
                )
            elif training_config.train_type == "unet_text_to_image":
                log_validation_train_unet_text_to_image(
                    tokenizer_one=tokenizer_one,
                    text_encoder_one=text_encoder_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    sigmas=sigmas.to(unet.module.dtype),
                    unet=unet,
                    num_validation_images=training_config.num_validation_images,
                    validation_prompts=training_config.validation_prompts,
                    timesteps=validation_timesteps,
                    training_step=training_step,
                )
            elif training_config.train_type == "unet_inpainting_cross_attention_conditioning":
                log_validation_train_unet_inpainting_cross_attention_conditioning(
                    tokenizer_one=tokenizer_one,
                    text_encoder_one=text_encoder_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_two=text_encoder_two,
                    vae=vae,
                    sigmas=sigmas.to(unet.module.dtype),
                    unet=unet,
                    num_validation_images=training_config.num_validation_images,
                    validation_prompts=training_config.validation_prompts,
                    validation_images=training_config.validation_images,
                    timesteps=validation_timesteps,
                    training_step=training_step,
                )
            else:
                assert False

        if dist.get_rank() == 0:
            loss = accumulated_loss.item()
            lr = lr_scheduler.get_last_lr()[0]
            logger.info(f"Step {training_step}: loss={loss}, lr={lr}")
            if training_config.log_to_wandb:
                wandb.log({"loss": loss, "lr": lr}, step=training_step)

    dist.barrier()

    if dist.get_rank() == 0:
        if training_config.controlnet is not None:
            save_models_train_controlnet(training_config.output_dir, optimizer=optimizer, controlnet=controlnet)
        elif training_config.train_type == "ema_unet_inpainting":
            save_models_train_ema_unet_inpainting(training_config.output_dir, optimizer=optimizer, unet=unet, ema_unet=ema_unet)
        elif training_config.train_type == "unet_inpainting":
            save_models_train_unet_inpainting(training_config.output_dir, optimizer=optimizer, unet=unet)
        elif training_config.train_type == "unet_text_to_image":
            save_models_train_unet_text_to_image(training_config.output_dir, optimizer=optimizer, unet=unet)
        elif training_config.train_type == "unet_inpainting_cross_attention_conditioning":
            save_models_train_unet_inpainting_cross_attention_conditioning(training_config.output_dir, optimizer=optimizer, unet=unet)
        else:
            assert False


def make_save_checkpoint(output_dir, checkpoints_total_limit, training_step):
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

    return save_path


def init_train_controlnet(training_config, make_dataloader=True):
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

    unet = SDXLUNet.load_fp16(device=device)
    unet.requires_grad_(False)
    unet.eval()

    if training_config.controlnet_resume_from is None:
        controlnet = SDXLControlNet.from_unet(unet)
        controlnet.to(device)
    else:
        controlnet = SDXLControlNet.load(training_config.controlnet_resume_from, device=device)
    controlnet.train()
    controlnet.requires_grad_(True)
    controlnet = DDP(controlnet, device_ids=[device])

    parameters = [x for x in controlnet.module.parameters()]

    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer = bnb.optim.AdamW8bit(parameters, lr=training_config.learning_rate)
    else:
        optimizer = AdamW(parameters, lr=training_config.learning_rate)

    if training_config.optimizer_resume_from is not None:
        optimizer.load_state_dict(torch.load(training_config.optimizer_resume_from, map_location=torch.device(device)))

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    rv = dict(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        sigmas=sigmas,
        unet=unet,
        controlnet=controlnet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parameters=parameters,
    )

    if make_dataloader:
        # TODO - get rid of this
        module, fn = training_config.dataloader.split(".")
        dataloader_fn = getattr(importlib.import_module(module), fn)
        dataloader = dataloader_fn(training_config, tokenizer_one, tokenizer_two)
        rv["dataloader"] = dataloader

    return rv


def train_step_train_controlnet(text_encoder_one, text_encoder_two, vae, sigmas, unet, batch, controlnet, training_config: TrainingConfig):
    with torch.no_grad():
        micro_conditioning = batch["micro_conditioning"].to(device=unet.device)

        image = batch["image"].to(vae.device, dtype=vae.dtype)
        latents = vae.encode(image).to(dtype=unet.dtype)

        text_input_ids_one = batch["text_input_ids_one"].to(text_encoder_one.device)
        text_input_ids_two = batch["text_input_ids_two"].to(text_encoder_two.device)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, sigmas.numel(), (bsz,), device=unet.device)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)[:, None, None, None]

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

        conditioning_image = batch["conditioning_image"].to(controlnet.module.device)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
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
        )

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


@torch.no_grad()
def log_validation_train_controlnet(
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
    controlnet,
    training_step,
    timesteps=None,
    do_log=True,
):
    controlnet_ = controlnet.module
    controlnet_.eval()

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

    generator = torch.Generator(unet.device).manual_seed(0)

    output_images = []

    for formatted_validation_image, validation_prompt in zip(formatted_validation_images, validation_prompts):
        for _ in range(num_validation_images):
            x_0 = sdxl_diffusion_loop(
                prompts=validation_prompt,
                images=formatted_validation_image,
                unet=unet,
                tokenizer_one=tokenizer_one,
                text_encoder_one=text_encoder_one,
                tokenizer_two=tokenizer_two,
                text_encoder_two=text_encoder_two,
                controlnet=controlnet_,
                sigmas=sigmas,
                timesteps=timesteps,
                generator=generator,
            )

            x_0 = vae.decode(x_0.to(vae.dtype))
            x_0 = vae.output_tensor_to_pil(x_0)[0]

            output_images.append(wandb.Image(x_0, caption=validation_prompt))

    controlnet_.train()

    if do_log:
        assert training_step is not None

        wandb.log({"validation": output_images}, step=training_step)
        wandb.log({"validation_conditioning": conditioning_images}, step=training_step)

    return output_images, conditioning_images


def save_models_train_controlnet(save_path: str, optimizer, controlnet):
    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    except RuntimeError as err:
        # TODO - RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 2075490688 vs 2075490580
        logger.warning(f"failed to save optimizer {err}")

    if has_safetensors:
        safetensors.torch.save_file(controlnet.module.state_dict(), os.path.join(save_path, "controlnet.safetensors"))
    else:
        torch.save(controlnet.module.state_dict(), os.path.join(save_path, "controlnet.bin"))

    logger.info(f"Saved to {save_path}")


def init_train_ema_unet_inpainting(training_config, make_dataloader=True):
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

    if training_config.unet_resume_from is None:
        unet = SDXLUNetInpainting.load_fp32(device)
    else:
        unet = SDXLUNetInpainting.load(training_config.unet_resume_from, device=device)

    unet.train()
    unet.requires_grad_(True)
    unet = DDP(unet, device_ids=[device])

    parameters = [x for x in unet.module.parameters()]

    if training_config.ema_unet_resume_from is None:
        kwargs = {}
        if training_config.ema_decay is not None:
            kwargs["decay"] = training_config.ema_decay
        if training_config.ema_power is not None:
            kwargs["power"] = training_config.ema_power
        ema_unet = EMAModel(parameters, **kwargs)
    else:
        ema_unet_state_dict = torch.load(training_config.ema_unet_resume_from, map_location=torch.device(device))
        ema_unet = EMAModel([])
        ema_unet.load_state_dict(ema_unet_state_dict)

    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer = bnb.optim.AdamW8bit(parameters, lr=training_config.learning_rate)
    else:
        optimizer = AdamW(parameters, lr=training_config.learning_rate)

    if training_config.optimizer_resume_from is not None:
        optimizer.load_state_dict(torch.load(training_config.optimizer_resume_from, map_location=torch.device(device)))

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    rv = dict(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        sigmas=sigmas,
        unet=unet,
        ema_unet=ema_unet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parameters=parameters,
    )

    if make_dataloader:
        from data import wds_dataloader_unet_inpainting_hq_dataset

        dataloader = wds_dataloader_unet_inpainting_hq_dataset(training_config, tokenizer_one, tokenizer_two)
        rv["dataloader"] = dataloader

    return rv


def train_step_train_ema_unet_inpainting(text_encoder_one, text_encoder_two, vae, sigmas, unet, batch, training_config: TrainingConfig):
    with torch.no_grad():
        micro_conditioning = batch["micro_conditioning"].to(device=unet.module.device)

        image = batch["image"].to(vae.device, dtype=vae.dtype)
        latents = vae.encode(image).to(dtype=unet.module.dtype)

        text_input_ids_one = batch["text_input_ids_one"].to(text_encoder_one.device)
        text_input_ids_two = batch["text_input_ids_two"].to(text_encoder_two.device)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.module.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.module.dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, sigmas.numel(), (bsz,), device=unet.module.device)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)[:, None, None, None]

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

        conditioning_image = vae.encode(batch["conditioning_image"].to(device=vae.device, dtype=vae.dtype)).to(dtype=unet.module.dtype)

        conditioning_image_mask = F.interpolate(batch["conditioning_image_mask"], size=scaled_noisy_latents.shape[2:]).to(device=unet.module.device, dtype=unet.module.dtype)

        model_input = torch.concat([scaled_noisy_latents, conditioning_image_mask, conditioning_image], dim=1)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        model_pred = unet(
            x_t=model_input,
            t=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


@torch.no_grad()
def log_validation_train_ema_unet_inpainting(
    tokenizer_one: Tokenizer,
    text_encoder_one,
    tokenizer_two: Tokenizer,
    text_encoder_two,
    vae,
    sigmas,
    unet,
    ema_unet,
    validation_images,
    validation_prompts,
    num_validation_images,
    training_step,
    timesteps=None,
    do_log=True,
):
    unet_ = unet.module
    unet_.eval()

    ema_unet.store(unet_.parameters())
    ema_unet.copy_to(unet_.parameters())

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

        from data import get_unet_inpainting_conditioning_image

        conditioning_image = get_unet_inpainting_conditioning_image(validation_image)

        conditioning_image_ = vae.encode(conditioning_image["conditioning_image"][None, :, :, :].to(device=vae.device, dtype=vae.dtype)).to(device=unet_.device, dtype=unet_.dtype)
        conditioning_image_mask = F.interpolate(conditioning_image["conditioning_image_mask"][None, :, :, :], size=conditioning_image_.shape[2:]).to(device=unet.module.device, dtype=unet.module.dtype)
        conditioning_image_ = torch.concat([conditioning_image_mask, conditioning_image_], dim=1)

        formatted_validation_images.append(conditioning_image_)

        conditioning_images.append(wandb.Image(conditioning_image["conditioning_image_as_pil"]))

    generator = torch.Generator(unet_.device).manual_seed(0)

    output_images = []

    for formatted_validation_image, validation_prompt in zip(formatted_validation_images, validation_prompts):
        for _ in range(num_validation_images):
            encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
                text_encoder_one,
                text_encoder_two,
                torch.tensor(tokenizer_one.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_one.device),
                torch.tensor(tokenizer_two.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_two.device),
            )
            encoder_hidden_states = encoder_hidden_states.to(unet_.dtype)
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(unet_.dtype)

            micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet_.device)

            x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet_.dtype, device=unet_.device, generator=generator)
            x_T = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

            eps_theta = lambda *args, **kwargs: sdxl_eps_theta_unet_inpainting(
                *args,
                **kwargs,
                unet=unet_,
                encoder_hidden_states=encoder_hidden_states,
                pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                negative_encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                negative_pooled_encoder_hidden_states=torch.zeros_like(pooled_encoder_hidden_states),
                micro_conditioning=micro_conditioning,
                inpainting_conditioning=formatted_validation_image.to(dtype=unet_.dtype, device=unet_.device),
            )

            x_0 = euler_ode_solver(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

            x_0 = vae.decode(x_0.to(vae.dtype))
            x_0 = vae.output_tensor_to_pil(x_0)[0]

            output_images.append(wandb.Image(x_0, caption=validation_prompt))

    ema_unet.restore(unet_.parameters())
    unet_.train()

    if do_log:
        wandb.log({"validation": output_images}, step=training_step)
        wandb.log({"validation_conditioning": conditioning_images}, step=training_step)

    return output_images, conditioning_images


def save_models_train_ema_unet_inpainting(save_path: str, optimizer, unet, ema_unet):
    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    except RuntimeError as err:
        # TODO - RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 2075490688 vs 2075490580
        logger.warning(f"failed to save optimizer {err}")

    torch.save(ema_unet.state_dict(), os.path.join(save_path, "ema_unet.bin"))

    if has_safetensors:
        safetensors.torch.save_file(unet.module.state_dict(), os.path.join(save_path, "unet.safetensors"))
    else:
        torch.save(unet.module.state_dict(), os.path.join(save_path, "unet.bin"))

    logger.info(f"Saved to {save_path}")


def init_train_unet_inpainting(training_config, make_dataloader=True):
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

    if training_config.unet_resume_from is None:
        unet = SDXLUNetInpainting.load_fp32(device)
    else:
        unet = SDXLUNetInpainting.load(training_config.unet_resume_from, device=device)

    unet.train()
    unet.requires_grad_(True)
    unet = DDP(unet, device_ids=[device])

    parameters = [x for x in unet.module.parameters()]

    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer = bnb.optim.AdamW8bit(parameters, lr=training_config.learning_rate)
    else:
        optimizer = AdamW(parameters, lr=training_config.learning_rate)

    if training_config.optimizer_resume_from is not None:
        optimizer.load_state_dict(torch.load(training_config.optimizer_resume_from, map_location=torch.device(device)))

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    rv = dict(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        sigmas=sigmas,
        unet=unet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parameters=parameters,
    )

    if make_dataloader:
        from data import wds_dataloader_unet_inpainting_hq_dataset

        dataloader = wds_dataloader_unet_inpainting_hq_dataset(training_config, tokenizer_one, tokenizer_two)
        rv["dataloader"] = dataloader

    return rv


def train_step_train_unet_inpainting(text_encoder_one, text_encoder_two, vae, sigmas, unet, batch, training_config: TrainingConfig):
    with torch.no_grad():
        micro_conditioning = batch["micro_conditioning"].to(device=unet.module.device)

        image = batch["image"].to(vae.device, dtype=vae.dtype)
        latents = vae.encode(image).to(dtype=unet.module.dtype)

        text_input_ids_one = batch["text_input_ids_one"].to(text_encoder_one.device)
        text_input_ids_two = batch["text_input_ids_two"].to(text_encoder_two.device)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.module.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.module.dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, sigmas.numel(), (bsz,), device=unet.module.device)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)[:, None, None, None]

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

        conditioning_image = vae.encode(batch["conditioning_image"].to(device=vae.device, dtype=vae.dtype)).to(dtype=unet.module.dtype)

        conditioning_image_mask = F.interpolate(batch["conditioning_image_mask"], size=scaled_noisy_latents.shape[2:]).to(device=unet.module.device, dtype=unet.module.dtype)

        model_input = torch.concat([scaled_noisy_latents, conditioning_image_mask, conditioning_image], dim=1)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        model_pred = unet(
            x_t=model_input,
            t=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


@torch.no_grad()
def log_validation_train_unet_inpainting(
    tokenizer_one: Tokenizer,
    text_encoder_one,
    tokenizer_two: Tokenizer,
    text_encoder_two,
    vae,
    sigmas,
    unet,
    validation_images,
    validation_prompts,
    num_validation_images,
    training_step,
    timesteps=None,
    do_log=True,
):
    unet_ = unet.module
    unet_.eval()

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

        from data import get_unet_inpainting_conditioning_image

        conditioning_image = get_unet_inpainting_conditioning_image(validation_image)

        conditioning_image_ = vae.encode(conditioning_image["conditioning_image"][None, :, :, :].to(device=vae.device, dtype=vae.dtype)).to(device=unet_.device, dtype=unet_.dtype)
        conditioning_image_mask = F.interpolate(conditioning_image["conditioning_image_mask"][None, :, :, :], size=conditioning_image_.shape[2:]).to(device=unet.module.device, dtype=unet.module.dtype)
        conditioning_image_ = torch.concat([conditioning_image_mask, conditioning_image_], dim=1)

        formatted_validation_images.append(conditioning_image_)

        conditioning_images.append(wandb.Image(conditioning_image["conditioning_image_as_pil"]))

    generator = torch.Generator(unet_.device).manual_seed(0)

    output_images = []

    for formatted_validation_image, validation_prompt in zip(formatted_validation_images, validation_prompts):
        for _ in range(num_validation_images):
            encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
                text_encoder_one,
                text_encoder_two,
                torch.tensor(tokenizer_one.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_one.device),
                torch.tensor(tokenizer_two.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_two.device),
            )
            encoder_hidden_states = encoder_hidden_states.to(unet_.dtype)
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(unet_.dtype)

            micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet_.device)

            x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet_.dtype, device=unet_.device, generator=generator)
            x_T = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

            eps_theta = lambda *args, **kwargs: sdxl_eps_theta_unet_inpainting(
                *args,
                **kwargs,
                unet=unet_,
                encoder_hidden_states=encoder_hidden_states,
                pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                negative_encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                negative_pooled_encoder_hidden_states=torch.zeros_like(pooled_encoder_hidden_states),
                micro_conditioning=micro_conditioning,
                inpainting_conditioning=formatted_validation_image.to(dtype=unet_.dtype, device=unet_.device),
            )

            x_0 = euler_ode_solver(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

            x_0 = vae.decode(x_0.to(vae.dtype))
            x_0 = vae.output_tensor_to_pil(x_0)[0]

            output_images.append(wandb.Image(x_0, caption=validation_prompt))

    unet_.train()

    if do_log:
        wandb.log({"validation": output_images}, step=training_step)
        wandb.log({"validation_conditioning": conditioning_images}, step=training_step)

    return output_images, conditioning_images


def save_models_train_unet_inpainting(save_path: str, optimizer, unet):
    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    except RuntimeError as err:
        # TODO - RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 2075490688 vs 2075490580
        logger.warning(f"failed to save optimizer {err}")

    if has_safetensors:
        safetensors.torch.save_file(unet.module.state_dict(), os.path.join(save_path, "unet.safetensors"))
    else:
        torch.save(unet.module.state_dict(), os.path.join(save_path, "unet.bin"))

    logger.info(f"Saved to {save_path}")


def init_train_unet_text_to_image(training_config, make_dataloader=True):
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

    if training_config.unet_resume_from is None:
        unet = SDXLUNet.load_fp32(device)
    else:
        unet = SDXLUNet.load(training_config.unet_resume_from, device=device)

    unet.train()
    unet.requires_grad_(True)
    unet = DDP(unet, device_ids=[device])

    parameters = [x for x in unet.module.parameters()]

    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer = bnb.optim.AdamW8bit(parameters, lr=training_config.learning_rate)
    else:
        optimizer = AdamW(parameters, lr=training_config.learning_rate)

    if training_config.optimizer_resume_from is not None:
        optimizer.load_state_dict(torch.load(training_config.optimizer_resume_from, map_location=torch.device(device)))

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    rv = dict(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        sigmas=sigmas,
        unet=unet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parameters=parameters,
    )

    if make_dataloader:
        from data import wds_dataloader_unet_text_to_image_hq_dataset

        dataloader = wds_dataloader_unet_text_to_image_hq_dataset(training_config, tokenizer_one, tokenizer_two)
        rv["dataloader"] = dataloader

    return rv


def train_step_train_unet_text_to_image(text_encoder_one, text_encoder_two, vae, sigmas, unet, batch, training_config: TrainingConfig):
    with torch.no_grad():
        micro_conditioning = batch["micro_conditioning"].to(device=unet.module.device)

        image = batch["image"].to(vae.device, dtype=vae.dtype)
        latents = vae.encode(image).to(dtype=unet.module.dtype)

        text_input_ids_one = batch["text_input_ids_one"].to(text_encoder_one.device)
        text_input_ids_two = batch["text_input_ids_two"].to(text_encoder_two.device)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.module.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.module.dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, sigmas.numel(), (bsz,), device=unet.module.device)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)[:, None, None, None]

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        model_pred = unet(
            x_t=scaled_noisy_latents,
            t=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        )

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


@torch.no_grad()
def log_validation_train_unet_text_to_image(
    tokenizer_one: Tokenizer,
    text_encoder_one,
    tokenizer_two: Tokenizer,
    text_encoder_two,
    vae,
    sigmas,
    unet,
    validation_prompts,
    num_validation_images,
    training_step,
    timesteps=None,
    do_log=True,
):
    unet_ = unet.module
    unet_.eval()

    generator = torch.Generator(unet_.device).manual_seed(0)

    output_images = []

    for validation_prompt in validation_prompts:
        for _ in range(num_validation_images):
            encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
                text_encoder_one,
                text_encoder_two,
                torch.tensor(tokenizer_one.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_one.device),
                torch.tensor(tokenizer_two.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_two.device),
            )
            encoder_hidden_states = encoder_hidden_states.to(unet_.dtype)
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(unet_.dtype)

            micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet_.device)

            x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet_.dtype, device=unet_.device, generator=generator)
            x_T = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

            eps_theta = lambda *args, **kwargs: sdxl_eps_theta_(
                *args,
                **kwargs,
                unet=unet_,
                encoder_hidden_states=encoder_hidden_states,
                pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                negative_encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                negative_pooled_encoder_hidden_states=torch.zeros_like(pooled_encoder_hidden_states),
                micro_conditioning=micro_conditioning,
            )

            x_0 = euler_ode_solver(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

            x_0 = vae.decode(x_0.to(vae.dtype))
            x_0 = vae.output_tensor_to_pil(x_0)[0]

            output_images.append(wandb.Image(x_0, caption=validation_prompt))

    unet_.train()

    if do_log:
        wandb.log({"validation": output_images}, step=training_step)

    return output_images


def save_models_train_unet_text_to_image(save_path: str, optimizer, unet):
    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    except RuntimeError as err:
        # TODO - RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 2075490688 vs 2075490580
        logger.warning(f"failed to save optimizer {err}")

    if has_safetensors:
        safetensors.torch.save_file(unet.module.state_dict(), os.path.join(save_path, "unet.safetensors"))
    else:
        torch.save(unet.module.state_dict(), os.path.join(save_path, "unet.bin"))

    logger.info(f"Saved to {save_path}")


def init_train_unet_inpainting_cross_attention_conditioning(training_config, make_dataloader=True):
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

    if training_config.unet_resume_from is None:
        unet = SDXLUNetInpaintingCrossAttentionConditioning.load_fp32(device)
    else:
        unet = SDXLUNetInpaintingCrossAttentionConditioning.load(training_config.unet_resume_from, device=device)

    unet.gradient_checkpointing = training_config.gradient_checkpointing
    set_attention_checkpoint_kv(training_config.gradient_checkpointing)

    unet.train()
    unet.requires_grad_(True)
    unet = DDP(unet, device_ids=[device])

    parameters = [x for x in unet.module.parameters()]

    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer = bnb.optim.AdamW8bit(parameters, lr=training_config.learning_rate)
    else:
        optimizer = AdamW(parameters, lr=training_config.learning_rate)

    if training_config.optimizer_resume_from is not None:
        optimizer_sd = torch.load(training_config.optimizer_resume_from, map_location=torch.device(device))
        optimizer_sd["param_groups"][0]["initial_lr"] = training_config.learning_rate
        optimizer_sd["param_groups"][0]["lr"] = training_config.learning_rate
        optimizer.load_state_dict(optimizer_sd)

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    rv = dict(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        sigmas=sigmas,
        unet=unet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parameters=parameters,
    )

    if make_dataloader:
        from data import wds_dataloader_unet_inpainting_hq_dataset

        dataloader = wds_dataloader_unet_inpainting_hq_dataset(training_config, tokenizer_one, tokenizer_two)
        rv["dataloader"] = dataloader

    return rv


def train_step_train_unet_inpainting_cross_attention_conditioning(text_encoder_one, text_encoder_two, vae, sigmas, unet, batch, training_config: TrainingConfig):
    with torch.no_grad():
        micro_conditioning = batch["micro_conditioning"].to(device=unet.module.device)

        image = batch["image"].to(vae.device, dtype=vae.dtype)
        latents = vae.encode(image).to(dtype=unet.module.dtype)

        text_input_ids_one = batch["text_input_ids_one"].to(text_encoder_one.device)
        text_input_ids_two = batch["text_input_ids_two"].to(text_encoder_two.device)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet.module.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.module.dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, sigmas.numel(), (bsz,), device=unet.module.device)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)[:, None, None, None]

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

        conditioning_image = vae.encode(batch["conditioning_image"].to(device=vae.device, dtype=vae.dtype)).to(dtype=unet.module.dtype)

        conditioning_image_mask = F.interpolate(batch["conditioning_image_mask"], size=scaled_noisy_latents.shape[2:]).to(device=unet.module.device, dtype=unet.module.dtype)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        model_pred = unet(
            x_t=scaled_noisy_latents,
            t=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            conditioning_image=conditioning_image,
            conditioning_image_mask=conditioning_image_mask,
        )

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


@torch.no_grad()
def log_validation_train_unet_inpainting_cross_attention_conditioning(
    tokenizer_one: Tokenizer,
    text_encoder_one,
    tokenizer_two: Tokenizer,
    text_encoder_two,
    vae,
    sigmas,
    unet,
    validation_images,
    validation_prompts,
    num_validation_images,
    training_step,
    timesteps=None,
    do_log=True,
):
    unet_ = unet.module
    unet_.eval()

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

        from data import get_unet_inpainting_conditioning_image

        conditioning_image = get_unet_inpainting_conditioning_image(validation_image)

        conditioning_image_ = vae.encode(conditioning_image["conditioning_image"][None, :, :, :].to(device=vae.device, dtype=vae.dtype)).to(device=unet_.device, dtype=unet_.dtype)

        conditioning_image_mask = F.interpolate(conditioning_image["conditioning_image_mask"][None, :, :, :], size=conditioning_image_.shape[2:]).to(device=unet.module.device, dtype=unet.module.dtype)

        formatted_validation_images.append((conditioning_image_, conditioning_image_mask))

        conditioning_images.append(wandb.Image(conditioning_image["conditioning_image_as_pil"]))

    generator = torch.Generator(unet_.device).manual_seed(0)

    output_images = []

    for formatted_validation_image, validation_prompt in zip(formatted_validation_images, validation_prompts):
        for _ in range(num_validation_images):
            encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
                text_encoder_one,
                text_encoder_two,
                torch.tensor(tokenizer_one.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_one.device),
                torch.tensor(tokenizer_two.encode(validation_prompt).ids, dtype=torch.long, device=text_encoder_two.device),
            )
            encoder_hidden_states = encoder_hidden_states.to(unet_.dtype)
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(unet_.dtype)

            micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet_.device)

            x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet_.dtype, device=unet_.device, generator=generator)
            x_T = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

            conditioning_image, conditioning_image_mask = formatted_validation_image

            eps_theta = lambda *args, **kwargs: sdxl_eps_theta_unet_inpainting_cross_attention_conditioning(
                *args,
                **kwargs,
                unet=unet_,
                encoder_hidden_states=encoder_hidden_states,
                pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                negative_encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                negative_pooled_encoder_hidden_states=torch.zeros_like(pooled_encoder_hidden_states),
                micro_conditioning=micro_conditioning,
                conditioning_image=conditioning_image.to(dtype=unet_.dtype, device=unet_.device),
                conditioning_image_mask=conditioning_image_mask.to(dtype=unet_.dtype, device=unet_.device),
            )

            x_0 = euler_ode_solver(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

            x_0 = vae.decode(x_0.to(vae.dtype))
            x_0 = vae.output_tensor_to_pil(x_0)[0]

            output_images.append(wandb.Image(x_0, caption=validation_prompt))

    unet_.train()

    if do_log:
        wandb.log({"validation": output_images}, step=training_step)
        wandb.log({"validation_conditioning": conditioning_images}, step=training_step)

    return output_images, conditioning_images


def save_models_train_unet_inpainting_cross_attention_conditioning(save_path: str, optimizer, unet):
    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    except RuntimeError as err:
        # TODO - RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 2075490688 vs 2075490580
        logger.warning(f"failed to save optimizer {err}")

    if has_safetensors:
        safetensors.torch.save_file(unet.module.state_dict(), os.path.join(save_path, "unet.safetensors"))
    else:
        torch.save(unet.module.state_dict(), os.path.join(save_path, "unet.bin"))

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
