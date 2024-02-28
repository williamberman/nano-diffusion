import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import (AutoTokenizer, CLIPTextModel,
                          CLIPTextModelWithProjection)

device = "cuda"

tokenizer_one = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer",
)
tokenizer_two = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer_2",
)
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
).to(device=device, dtype=torch.float16)
text_encoder_one = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="text_encoder",
).to(device=device, dtype=torch.float16)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="text_encoder_2",
).to(device=device, dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(
    device=device, dtype=torch.float16
)

nsteps = 20
prompts = ["A cat", "A dog"]
cfg = 5

with torch.no_grad():
    betas = (
        torch.linspace(
            0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float32, device=device
        )
        ** 2
    )
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    timesteps = torch.linspace(999, -1, nsteps + 1, dtype=torch.long, device=device)[
        :-1
    ]
    micro_conditioning = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=device
    )

    tokenized_one = tokenizer_one(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    tokenized_two = tokenizer_two(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    _, _, hidden_states_one = text_encoder_one(
        tokenized_one.to(device=text_encoder_one.device),
        output_hidden_states=True,
        return_dict=False,
    )
    pooled_prompt_embeds_two, _, hidden_states_two = text_encoder_two(
        tokenized_two.to(device=text_encoder_two.device),
        output_hidden_states=True,
        return_dict=False,
    )
    prompt_embeds = torch.concat([hidden_states_one[-2], hidden_states_two[-2]], dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds_two.reshape(
        pooled_prompt_embeds_two.shape[0], -1
    )

    if cfg > 1:
        prompt_embeds = torch.concat(
            [prompt_embeds, torch.zeros_like(prompt_embeds)], dim=0
        )
        pooled_prompt_embeds = torch.concat(
            [pooled_prompt_embeds, torch.zeros_like(pooled_prompt_embeds)], dim=0
        )

    x_t = (
        torch.randn((len(prompts), 4, 128, 128), dtype=torch.float16, device=device)
        * sigmas[-1]
    )
    x_ts = [x_t]

    for i, t in tqdm(enumerate(timesteps)):
        x_t_input = torch.concat([x_t, x_t], dim=0) if cfg > 1 else x_t
        x_t_input = x_t_input / ((sigmas[t] ** 2 + 1) ** 0.5)

        dx_by_dsigma = unet(
            x_t_input,
            t.unsqueeze(0),
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": micro_conditioning.repeat(x_t_input.shape[0], 1),
            },
        ).sample

        if cfg > 1:
            cond, uncond = dx_by_dsigma.chunk(2)
            dx_by_dsigma = uncond + cfg * (cond - uncond)

        if i == timesteps.shape[0] - 1:
            dsigma = -sigmas[t]
        else:
            dsigma = sigmas[timesteps[i + 1]] - sigmas[t]

        x_t = x_t + dx_by_dsigma * dsigma
        x_ts.append(x_t)

    x_ts_pils = []
    for x_t in x_ts:
        x_ts_pils.append(
            [
                Image.fromarray(x)
                for x in (
                    (vae.decode(x_t / vae.config.scaling_factor).sample * 0.5 + 0.5)
                    * 255
                )
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            ]
        )

import mediapy as media

for prompt_idx, prompt in enumerate(prompts):
    prompt_ims = {}
    for t_idx, t in enumerate(timesteps):
        prompt_ims[t.item()] = x_ts_pils[t_idx][prompt_idx]
    prompt_ims["unnoised"] = x_ts_pils[-1][prompt_idx]
    media.show_images(prompt_ims, ylabel=prompt)
