# Nano Diffusion

Nano Diffusion is a small (< 3k loc) and self contained implementation of diffusion models.
Its primary focus is Stable Diffusion XL inference and training including support in both for controlnet and t2i adapter auxiliary networks.

It is self contained, with few dependencies. 

Mandatory inference dependencies: torch, numpy, tokenizers
Optional inference dependencies: huggingface_hub, safetensors, tqdm, xformers

Mandatory training dependencies: torch, numpy, tokenizers, huggingface_hub, wandb
Optional training dependencies: tqdm, xformers

TODO - is PIL a dependency or is it baked into python?

## Inference

NOTE: these are a wip and likely going to change -- i.e. I'm probably going to only allow passing text embeddings to `sdxl_diffusion_loop`

### Dependencies

Install pytorch TODO pytorch installation instructions - note torch 2.0

Install other dependencies

```sh
pip install numpy tokenizers
```

These examples will all use the optional huggingface_hub dependency to download models

```sh
pip install huggingface_hub
```

I strongly recommend installing safetensors. However, it is optional

```sh
pip install safetensors
```

### Basic inference

```python
from models import make_clip_tokenizer_from_hub, SDXLCLIPOne, SDXLCLIPTwo, SDXLVae, SDXLUNet

device = 'cuda'

# downloads vocab and merge files from hub and instantiate a `tokenizers` tokenizer
tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

# downloads the canonical fp32 sdxl checkpoint for the model component
text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
unet = SDXLUNet.load_fp32(device=device)

# runs the diffusion process in reverse to sample vae latents
images_tensor = sdxl_diffusion_loop(
    "horse",
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
)

# converts the image tensors from vae latents to PIL images
image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

image_pils[0].save("out.png")
```

### Batched inference

```python
from models import make_clip_tokenizer_from_hub, SDXLCLIPOne, SDXLCLIPTwo, SDXLVae, SDXLUNet

device = 'cuda'

tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
unet = SDXLUNet.load_fp32(device=device)

images_tensor = sdxl_diffusion_loop(
    # Prompts accepts both a list of strings and a string
    ["horse", "cow", "dog"],
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
)

image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

for i, image_pil in enumerate(image_pils):
    image_pil.save(f"out_{i}.png")
```

### FP16 Inference

Load models in fp16 and use the same `sdxl_diffusion_loop` function. 
`{SDXLCLIPOne,SDXLCLIPTwo,SDXLUNet}.load_fp16` are helper methods that
will download fp16 weights of the canonical sdxl models. The checkpoint downloaded 
by `SDXLVae.load_fp16_fix` has weights in fp32 and so must be manually cast.

```python
from models import make_clip_tokenizer_from_hub, SDXLCLIPOne, SDXLCLIPTwo, SDXLVae, SDXLUNet
import torch

device = 'cuda'

tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

# downloads the canonical fp16 sdxl checkpoint for the model component
text_encoder_one = SDXLCLIPOne.load_fp16(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp16(device=device)

# The checkpoint downloaded by `SDXLVae.load_fp16_fix` has weights in fp32 and 
# must be manually cast.
vae = SDXLVae.load_fp16_fix(device=device)
vae.to(torch.float16)

unet = SDXLUNet.load_fp16(device=device)

images_tensor = sdxl_diffusion_loop(
    "horse",
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
)

image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

image_pils[0].save("out.png")
```

### Deterministic RNG

```python
from models import make_clip_tokenizer_from_hub, SDXLCLIPOne, SDXLCLIPTwo, SDXLVae, SDXLUNet
import torch

device = 'cuda'

tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
unet = SDXLUNet.load_fp32(device=device)

images_tensor = sdxl_diffusion_loop(
    "horse",
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
    # Pass a generator for deterministic RNG
    generator=torch.Generator(device).manual_seed(0)
)

image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

image_pils[0].save("out.png")
```

### Set a different incremental sampling algorithm i.e. higher order ode solvers

```python
from models import make_clip_tokenizer_from_hub, SDXLCLIPOne, SDXLCLIPTwo, SDXLVae, SDXLUNet
from diffusion import heun_ode_solver

device = 'cuda'

tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
unet = SDXLUNet.load_fp32(device=device)

images_tensor = sdxl_diffusion_loop(
    "horse",
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
    # pass the alternative sampling algorithm
    sampler=heun_ode_solver
)

image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

image_pils[0].save("out.png")
```

### Set different timesteps

```python
from models import make_clip_tokenizer_from_hub, SDXLCLIPOne, SDXLCLIPTwo, SDXLVae, SDXLUNet
from diffusion import make_sigmas

device = 'cuda'

tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
unet = SDXLUNet.load_fp32(device=device)

# Timesteps must be a tensor of indices into sigmas. They should be in increasing order
sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
timesteps = torch.linspace(0, sigmas.numel() - 1, 20, dtype=torch.long, device=unet.device)

images_tensor = sdxl_diffusion_loop(
    "horse",
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
    sigmas=sigmas,
    timesteps=timesteps,
)

image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

image_pils[0].save("out.png")
```

### Inference with controlnet

TODO document opencv download

```python
import cv2
from huggingface_hub import hf_hub_download

device = 'cuda'

tokenizer_one = make_clip_tokenizer_one_from_hub()
tokenizer_two = make_clip_tokenizer_two_from_hub()

text_encoder_one = SDXLCLIPOne.load_fp32(device=device)
text_encoder_two = SDXLCLIPTwo.load_fp32(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
unet = SDXLUNet.load_fp32(device=device)

controlnet = SDXLControlNet.load(hf_hub_download("diffusers/controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors"), device=device)

image = Image.open(hf_hub_download("williamberman/misc", "bright_room_with_chair.png", repo_type="dataset")).convert("RGB").resize((1024, 1024))
image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
image = image[None, :, :, :].to(device=device, dtype=controlnet.dtype)

images_tensor = sdxl_diffusion_loop(
    "a beautiful room",
    unet=unet,
    tokenizer_one=tokenizer_one,
    text_encoder_one=text_encoder_one,
    tokenizer_two=tokenizer_two,
    text_encoder_two=text_encoder_two,
    controlnet=controlnet,
    images=image,
)

image_pils = vae.output_tensor_to_pil(vae.decode(images_tensor))

image_pils[0].save("out.png")
```

### Inference with t2i adapter

TODO

```python
```

## Training

`train.py` is a training loop written assuming targetting cuda and ddp. Because it assumes ddp, 
the script should always be launched with torchrun even if running on a single GPU.

Training config is placed in a yaml file pointed to by the env var NANO_DIFFUSION_TRAINING_CONFIG or passed via the cli flag `--config_path`.

`train.slurm` is a slurm driver script to launch `train.py` on multiple nodes on a slurm cluster. It works on the cluster I use, but ymmv.

TODO - how to document data

### Dependencies

Install pytorch TODO pytorch installation instructions - note torch 2.0

Install other dependencies

```sh
pip install numpy tokenizers huggingface_hub wandb
```

I strongly recommend installing safetensors. However, it is optional

```sh
pip install safetensors
```

### Single Machine, Single GPU

```sh
NANO_DIFFUSION_TRAINING_CONFIG="<path to config file>" \
    torchrun \
        --standalone \
        --nproc_per_node=1 \
        train.py
```

or

```sh
torchrun \
    --standalone \
    --nproc_per_node=1 \
    train.py \
    --config_path "<path to config file>"
```

### Single Machine, Multiple GPUs

```sh
NANO_DIFFUSION_TRAINING_CONFIG="<path to config file>" \
    torchrun \
        --standalone \
        --nproc_per_node=<number of gpus> \
        train.py
```

or

```sh
torchrun \
    --standalone \
    --nproc_per_node=<number of gpus> \
    train.py \
    --config_path "<path to config file>"
```

### Multiple Machines, Multiple GPUs

```sh
NANO_DIFFUSION_TRAINING_CONFIG="<path to config file>" \
    sbatch \
        --nodes=<number of nodes> \
        --output=<log file> \
        train.slurm
```
