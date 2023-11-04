from typing import List, Union

import torch

from models import sdxl_text_conditioning

_with_tqdm = False


def set_with_tqdm(it: bool):
    global _with_tqdm

    _with_tqdm = it


@torch.no_grad()
def make_sigmas(beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000, device=None):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32, device=device) ** 2

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # TODO - would be nice to use a direct expression for this
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

    return sigmas


@torch.no_grad()
def euler_ode_solver(eps_theta, timesteps, sigmas, x_T):
    x_t_i = x_T

    iter_over = range(len(timesteps) - 1, -1, -1)

    if _with_tqdm:
        from tqdm import tqdm

        iter_over = tqdm(iter_over)

    for i in iter_over:
        t_i = timesteps[i].unsqueeze(0)
        sigma_t_i = sigmas[t_i]

        dx_by_dsigma = eps_theta(x_t=x_t_i, t=t_i, sigma=sigma_t_i)

        if i == 0:
            dsigma = -sigma_t_i
        else:
            dsigma = sigmas[timesteps[i - 1].unsqueeze(0)] - sigma_t_i

        dx = dx_by_dsigma * dsigma

        x_t_i = x_t_i + dx

    return x_t_i


@torch.no_grad()
def heun_ode_solver(eps_theta, timesteps, sigmas, x_T):
    x_t_i = x_T

    iter_over = range(len(timesteps) - 1, -1, -1)

    if _with_tqdm:
        from tqdm import tqdm

        iter_over = tqdm(iter_over)

    for i in iter_over:
        t_i = timesteps[i].unsqueeze(0)
        sigma_t_i = sigmas[t_i]

        if i == 0:
            dsigma = -sigma_t_i
            dx_by_dsigma = eps_theta(x_t=x_t_i, t=t_i, sigma=sigma_t_i)
        else:
            t_i_minus_1 = timesteps[i - 1].unsqueeze(0)
            sigma_t_i_minus_1 = sigmas[t_i_minus_1]

            dsigma = sigma_t_i_minus_1 - sigma_t_i

            dx_by_dsigma_1 = eps_theta(x_t=x_t_i, t=t_i, sigma=sigma_t_i)
            dx_by_dsigma_2 = eps_theta(x_t=x_t_i + dx_by_dsigma_1 * dsigma, t=t_i_minus_1, sigma=sigma_t_i_minus_1)
            dx_by_dsigma = (dx_by_dsigma_1 + dx_by_dsigma_2) / 2

        x_t_i = x_t_i + dx_by_dsigma * dsigma

    return x_t_i


@torch.no_grad()
def rk4_ode_solver(eps_theta, timesteps, sigmas, x_T):
    x_t_i = x_T

    iter_over = range(len(timesteps) - 1, -1, -1)

    if _with_tqdm:
        from tqdm import tqdm

        iter_over = tqdm(iter_over)

    for i in iter_over:
        t_i = timesteps[i].unsqueeze(0)
        sigma_t_i = sigmas[t_i]

        if i == 0:
            dsigma = -sigma_t_i
            dx_by_dsigma = eps_theta(x_t=x_t_i, t=t_i, sigma=sigma_t_i)
        else:
            t_i_minus_1 = timesteps[i - 1].unsqueeze(0)
            sigma_t_i_minus_1 = sigmas[t_i_minus_1]

            dsigma = sigma_t_i_minus_1 - sigma_t_i

            dx_by_dsigma_1 = eps_theta(x_t=x_t_i, t=t_i, sigma=sigma_t_i)

            t_mid = t_i + (1 / 2 * (t_i_minus_1 - t_i)).round().to(dtype=torch.long)
            sigma_mid = sigmas[t_mid]
            dsigma_mid = sigma_mid - sigma_t_i

            dx_by_dsigma_2 = eps_theta(x_t=x_t_i + dx_by_dsigma_1 * dsigma_mid, t=t_mid, sigma=sigma_mid)
            dx_by_dsigma_3 = eps_theta(x_t=x_t_i + dx_by_dsigma_2 * dsigma_mid, t=t_mid, sigma=sigma_mid)

            dx_by_dsigma_4 = eps_theta(x_t=x_t_i + dx_by_dsigma_3 * dsigma, t=t_i_minus_1, sigma=sigma_t_i_minus_1)

            dx_by_dsigma = 1 / 6 * dx_by_dsigma_1 + 1 / 3 * dx_by_dsigma_2 + 1 / 3 * dx_by_dsigma_3 + 1 / 6 * dx_by_dsigma_4

        x_t_i = x_t_i + dx_by_dsigma * dsigma

    return x_t_i


@torch.no_grad()
def sdxl_diffusion_loop(
    prompts: Union[str, List[str]] = None,
    unet=None,
    tokenizer_one=None,
    text_encoder_one=None,
    tokenizer_two=None,
    text_encoder_two=None,
    images=None,
    controlnet=None,
    adapter=None,
    sigmas=None,
    timesteps=None,
    x_T=None,
    micro_conditioning=None,
    guidance_scale=5.0,
    generator=None,
    negative_prompts=None,
    sampler=euler_ode_solver,
    encoder_hidden_states=None,
    pooled_encoder_hidden_states=None,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    if prompts is not None:
        batch_size = len(prompts)
    else:
        batch_size = encoder_hidden_states.shape[0]

    if negative_prompts is not None and guidance_scale > 1.0:
        prompts += negative_prompts

    if encoder_hidden_states is None:
        assert pooled_encoder_hidden_states is None

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
            text_encoder_one,
            text_encoder_two,
            torch.tensor([x.ids for x in tokenizer_one.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
            torch.tensor([x.ids for x in tokenizer_two.encode_batch(prompts)], dtype=torch.long, device=text_encoder_one.device),
        )
        encoder_hidden_states = encoder_hidden_states.to(unet.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(unet.dtype)
    else:
        assert pooled_encoder_hidden_states is not None

    if guidance_scale > 1.0:
        if negative_prompts is None:
            negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            negative_pooled_encoder_hidden_states = torch.zeros_like(pooled_encoder_hidden_states)
        else:
            encoder_hidden_states, negative_encoder_hidden_states = torch.chunk(encoder_hidden_states, 2)
            pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states = torch.chunk(pooled_encoder_hidden_states, 2)
    else:
        negative_encoder_hidden_states = None
        negative_pooled_encoder_hidden_states = None

    if sigmas is None:
        sigmas = make_sigmas(device=unet.device)

    if timesteps is None:
        timesteps = torch.linspace(0, sigmas.numel() - 1, 50, dtype=torch.long, device=unet.device)

    if x_T is None:
        x_T = torch.randn((batch_size, 4, 1024 // 8, 1024 // 8), dtype=unet.dtype, device=unet.device, generator=generator)
        x_T = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

    if micro_conditioning is None:
        micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet.device)
        micro_conditioning = micro_conditioning.expand(batch_size, -1)

    if adapter is not None:
        add_to_down_block_outputs = adapter(images.to(dtype=adapter.dtype, device=adapter.device))
        add_to_down_block_outputs = [x.to(unet.dtype) for x in add_to_down_block_outputs]
    else:
        add_to_down_block_outputs = None

    if controlnet is not None:
        controlnet_cond = images.to(dtype=controlnet.dtype, device=controlnet.device)
    else:
        controlnet_cond = None

    eps_theta = lambda *args, **kwargs: sdxl_eps_theta(
        *args,
        **kwargs,
        unet=unet,
        encoder_hidden_states=encoder_hidden_states,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        negative_encoder_hidden_states=negative_encoder_hidden_states,
        negative_pooled_encoder_hidden_states=negative_pooled_encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        guidance_scale=guidance_scale,
        controlnet=controlnet,
        controlnet_cond=controlnet_cond,
        add_to_down_block_outputs=add_to_down_block_outputs,
    )

    x_0 = sampler(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

    return x_0


@torch.no_grad()
def sdxl_eps_theta(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    negative_encoder_hidden_states,
    negative_pooled_encoder_hidden_states,
    micro_conditioning,
    guidance_scale,
    controlnet=None,
    controlnet_cond=None,
    add_to_down_block_outputs=None,
):
    # TODO - how does this not effect the ode we are solving
    scaled_x_t = x_t / ((sigma**2 + 1) ** 0.5)

    if guidance_scale > 1.0:
        scaled_x_t = torch.concat([scaled_x_t, scaled_x_t])

        encoder_hidden_states = torch.concat((encoder_hidden_states, negative_encoder_hidden_states))
        pooled_encoder_hidden_states = torch.concat((pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states))

        micro_conditioning = torch.concat([micro_conditioning, micro_conditioning])

        if controlnet_cond is not None:
            controlnet_cond = torch.concat([controlnet_cond, controlnet_cond])

        if add_to_down_block_outputs is not None:
            add_to_down_block_outputs = [torch.concat([x, x]) for x in add_to_down_block_outputs]

    if controlnet is not None:
        controlnet_out = controlnet(
            x_t=scaled_x_t.to(controlnet.dtype),
            t=t,
            encoder_hidden_states=encoder_hidden_states.to(controlnet.dtype),
            micro_conditioning=micro_conditioning.to(controlnet.dtype),
            pooled_encoder_hidden_states=pooled_encoder_hidden_states.to(controlnet.dtype),
            controlnet_cond=controlnet_cond,
        )

        down_block_additional_residuals = [x.to(unet.dtype) for x in controlnet_out["down_block_res_samples"]]
        mid_block_additional_residual = controlnet_out["mid_block_res_sample"].to(unet.dtype)
    else:
        down_block_additional_residuals = None
        mid_block_additional_residual = None

    eps_hat = unet(
        x_t=scaled_x_t,
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        down_block_additional_residuals=down_block_additional_residuals,
        mid_block_additional_residual=mid_block_additional_residual,
        add_to_down_block_outputs=add_to_down_block_outputs,
    )

    if guidance_scale > 1.0:
        eps_hat, eps_hat_uncond = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat


@torch.no_grad()
def sdxl_eps_theta_(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    negative_encoder_hidden_states,
    negative_pooled_encoder_hidden_states,
    micro_conditioning,
    guidance_scale,
):
    # TODO - how does this not effect the ode we are solving
    scaled_x_t = x_t / ((sigma**2 + 1) ** 0.5)

    if guidance_scale > 1.0:
        scaled_x_t = torch.concat([scaled_x_t, scaled_x_t])

        encoder_hidden_states = torch.concat((encoder_hidden_states, negative_encoder_hidden_states))
        pooled_encoder_hidden_states = torch.concat((pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states))

        micro_conditioning = torch.concat([micro_conditioning, micro_conditioning])

    eps_hat = unet(
        x_t=scaled_x_t,
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
    )

    if guidance_scale > 1.0:
        eps_hat, eps_hat_uncond = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat


@torch.no_grad()
def sdxl_eps_theta_controlnet(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    negative_encoder_hidden_states,
    negative_pooled_encoder_hidden_states,
    micro_conditioning,
    guidance_scale,
    controlnet,
    controlnet_cond,
):
    # TODO - how does this not effect the ode we are solving
    scaled_x_t = x_t / ((sigma**2 + 1) ** 0.5)

    if guidance_scale > 1.0:
        scaled_x_t = torch.concat([scaled_x_t, scaled_x_t])

        encoder_hidden_states = torch.concat((encoder_hidden_states, negative_encoder_hidden_states))
        pooled_encoder_hidden_states = torch.concat((pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states))

        micro_conditioning = torch.concat([micro_conditioning, micro_conditioning])

        controlnet_cond = torch.concat([controlnet_cond, controlnet_cond])

    controlnet_out = controlnet(
        x_t=scaled_x_t.to(controlnet.dtype),
        t=t,
        encoder_hidden_states=encoder_hidden_states.to(controlnet.dtype),
        micro_conditioning=micro_conditioning.to(controlnet.dtype),
        pooled_encoder_hidden_states=pooled_encoder_hidden_states.to(controlnet.dtype),
        controlnet_cond=controlnet_cond,
    )

    down_block_additional_residuals = [x.to(unet.dtype) for x in controlnet_out["down_block_res_samples"]]
    mid_block_additional_residual = controlnet_out["mid_block_res_sample"].to(unet.dtype)

    eps_hat = unet(
        x_t=scaled_x_t,
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        down_block_additional_residuals=down_block_additional_residuals,
        mid_block_additional_residual=mid_block_additional_residual,
    )

    if guidance_scale > 1.0:
        eps_hat, eps_hat_uncond = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat


@torch.no_grad()
def sdxl_eps_theta_unet_inpainting(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    negative_encoder_hidden_states,
    negative_pooled_encoder_hidden_states,
    micro_conditioning,
    inpainting_conditioning,
    guidance_scale=5.0,
):
    # TODO - how does this not effect the ode we are solving
    scaled_x_t = x_t / ((sigma**2 + 1) ** 0.5)

    if guidance_scale > 1.0:
        scaled_x_t = torch.concat([scaled_x_t, scaled_x_t])

        encoder_hidden_states = torch.concat((encoder_hidden_states, negative_encoder_hidden_states))
        pooled_encoder_hidden_states = torch.concat((pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states))

        micro_conditioning = torch.concat([micro_conditioning, micro_conditioning])

        inpainting_conditioning = torch.concat([inpainting_conditioning, inpainting_conditioning])

    eps_hat = unet(
        x_t=torch.concat([scaled_x_t, inpainting_conditioning], dim=1),
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
    )

    if guidance_scale > 1.0:
        eps_hat, eps_hat_uncond = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat
