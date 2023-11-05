import math
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from tokenizers import (Regex, Tokenizer, decoders, normalizers,
                        pre_tokenizers, processors)
from tokenizers.models import BPE
from torch import nn

try:
    import safetensors.torch

    has_safetensors = True
except ImportError:
    has_safetensors = False


def load_file(load_from, device):
    if load_from.endswith(".safetensors"):
        if not has_safetensors:
            raise ImportError("safetensors is not installed")
        state_dict = safetensors.torch.load_file(load_from, device=device)
    else:
        state_dict = torch.load(load_from, map_location=device)

    if load_from.endswith(".safetensors"):
        if not has_safetensors:
            raise ImportError("safetensors is not installed")
        state_dict = safetensors.torch.load_file(load_from, device="cpu")
    else:
        state_dict = torch.load(load_from, map_location="cpu")

    return state_dict


class ModelUtils:
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def load(cls, load_from: str, device="cpu", strict=True):
        if version.parse(torch.__version__) >= version.parse("2.1"):
            state_dict = load_file(load_from, device=device)

            with torch.device("meta"):
                model = cls()

            model.load_state_dict(state_dict, assign=True, strict=strict)
        else:
            state_dict = load_file(load_from, device="cpu")

            with torch.device(device):
                model = cls()

            model.load_state_dict(state_dict, strict=strict)

            # HACK: this assumes all are the same dtype
            model.to(next(iter(state_dict.values())).dtype)

        return model


def make_clip_tokenizer(vocab_file, merges_file, pad_token_id):
    vocab, merges = BPE.read_file(vocab_file, merges_file)
    bos_token = "<|startoftext|>"
    unk_token = eos_token = "<|endoftext|>"
    bos_token_id = 49406
    eos_token_id = 49407

    tokenizer = Tokenizer(
        BPE(
            vocab=vocab,
            merges=merges,
            dropout=None,
            continuing_subword_prefix="",
            end_of_word_suffix="</w>",
            fuse_unk=False,
            unk_token=str(unk_token),
        )
    )

    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                behavior="removed",
                invert=True,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    # Hack to have a ByteLevel and TemplaceProcessor
    tokenizer.post_processor = processors.RobertaProcessing(
        sep=(eos_token, eos_token_id),
        cls=(bos_token, bos_token_id),
        add_prefix_space=False,
        trim_offsets=False,
    )

    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=pad_token_id)

    return tokenizer


def make_clip_tokenizer_one(vocab_file, merges_file):
    return make_clip_tokenizer(vocab_file, merges_file, 49407)


def make_clip_tokenizer_two(vocab_file, merges_file):
    return make_clip_tokenizer(vocab_file, merges_file, 0)


def make_clip_tokenizer_one_from_hub():
    from huggingface_hub import hf_hub_download

    vocab_file = hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "tokenizer/vocab.json")
    merges_file = hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "tokenizer/merges.txt")

    return make_clip_tokenizer_one(vocab_file, merges_file)


def make_clip_tokenizer_two_from_hub():
    from huggingface_hub import hf_hub_download

    vocab_file = hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "tokenizer/vocab.json")
    merges_file = hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "tokenizer/merges.txt")

    return make_clip_tokenizer_two(vocab_file, merges_file)


class CLIP(nn.Module, ModelUtils):
    def __init__(self, hidden_size, num_hidden_layers, act_fn, projection_dim=None):
        super().__init__()

        # fmt: off

        self.act_fn = act_fn

        self.register_buffer("position_ids", torch.arange(77).expand((1, -1)), persistent=False)

        self.text_model = nn.ModuleDict(dict(
            embeddings = nn.ModuleDict(dict(
                token_embedding=nn.Embedding(49408, hidden_size),
                position_embedding=nn.Embedding(77, hidden_size)
            )),

            encoder = nn.ModuleDict(dict(layers=nn.ModuleList([
                nn.ModuleDict(dict(
                    self_attn=CLIPAttention(hidden_size),
                    layer_norm1=nn.LayerNorm(hidden_size),
                    mlp=nn.ModuleDict(dict(fc1=nn.Linear(hidden_size, hidden_size*4), fc2=nn.Linear(hidden_size*4, hidden_size))),
                    layer_norm2=nn.LayerNorm(hidden_size),
                ))
                for _ in range(num_hidden_layers)
            ]))),

            final_layer_norm = nn.LayerNorm(hidden_size)
        ))

        if projection_dim is not None:
            self.text_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        else:
            self.text_projection = None

        # fmt: on

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.text_model["embeddings"]["token_embedding"](input_ids)
        position_embeddings = self.text_model["embeddings"]["position_embedding"](self.position_ids[:, : input_ids.shape[-1]])
        hidden_states = inputs_embeds + position_embeddings

        all_hidden_states = []

        for encoder_layer in self.text_model["encoder"]["layers"]:
            residual = hidden_states
            hidden_states = encoder_layer["layer_norm1"](hidden_states)
            hidden_states = encoder_layer["self_attn"](
                hidden_states=hidden_states,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = encoder_layer["layer_norm2"](hidden_states)
            hidden_states = encoder_layer["mlp"]["fc1"](hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = encoder_layer["mlp"]["fc2"](hidden_states)
            hidden_states = residual + hidden_states

            all_hidden_states.append(hidden_states)

        last_hidden_state = all_hidden_states[-1]
        last_hidden_state = self.text_model["final_layer_norm"](last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if self.text_projection is not None:
            pooled_output = self.text_projection(pooled_output)

        return pooled_output, all_hidden_states


class SDXLCLIPOne(CLIP):
    def __init__(self):
        super().__init__(hidden_size=768, num_hidden_layers=12, act_fn=quick_gelu)

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl clip from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "text_encoder/model.safetensors"), device=device)

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl clip from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "text_encoder/model.fp16.safetensors"), device=device)


class SDXLCLIPTwo(CLIP):
    def __init__(self):
        super().__init__(hidden_size=1280, num_hidden_layers=32, act_fn=F.gelu, projection_dim=1280)

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl clip from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "text_encoder_2/model.safetensors"), device=device)

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl clip from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "text_encoder_2/model.fp16.safetensors"), device=device)


def sdxl_text_conditioning(text_encoder_one: SDXLCLIPOne, text_encoder_two: SDXLCLIPTwo, text_input_ids_one, text_input_ids_two):
    _, hidden_states_one = text_encoder_one(text_input_ids_one)

    pooled_encoder_hidden_states, hidden_states_two = text_encoder_two(text_input_ids_two)

    encoder_hidden_states = torch.cat((hidden_states_one[-2], hidden_states_two[-2]), dim=-1)

    return encoder_hidden_states, pooled_encoder_hidden_states


class SDXLVae(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        self.encoder = nn.ModuleDict(dict(
            # 3 -> 128
            conv_in=nn.Conv2d(3, 128, kernel_size=3, padding=1),

            down_blocks=nn.ModuleList([
                # 128 -> 128
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(128, 128, eps=1e-6), ResnetBlock2D(128, 128, eps=1e-6)]),
                    downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(128, 128, kernel_size=3, stride=2)))]),
                )),
                # 128 -> 256
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(128, 256, eps=1e-6), ResnetBlock2D(256, 256, eps=1e-6)]),
                    downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(256, 256, kernel_size=3, stride=2)))]),
                )),
                # 256 -> 512
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(256, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
                    downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(512, 512, kernel_size=3, stride=2)))]),
                )),
                # 512 -> 512
                nn.ModuleDict(dict(resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]))),
            ]),

            # 512 -> 512
            mid_block=nn.ModuleDict(dict(
                attentions=nn.ModuleList([VaeMidBlockAttention(512)]),
                resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
            )),

            # 512 -> 8
            conv_norm_out=nn.GroupNorm(32, 512, eps=1e-06),
            conv_act=nn.SiLU(),
            conv_out=nn.Conv2d(512, 8, kernel_size=3, padding=1)
        ))

        # 8 -> 8
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)

        # 8 -> 4 from sampling mean and std

        # 4 -> 4
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1)

        self.decoder = nn.ModuleDict(dict(
            # 4 -> 512
            conv_in=nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # 512 -> 512
            mid_block=nn.ModuleDict(dict(
                attentions=nn.ModuleList([VaeMidBlockAttention(512)]),
                resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
            )),

            up_blocks=nn.ModuleList([
                # 512 -> 512
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
                    upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)))]),
                )),

                # 512 -> 512
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
                    upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)))]),
                )),

                # 512 -> 256
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(512, 256, eps=1e-6), ResnetBlock2D(256, 256, eps=1e-6), ResnetBlock2D(256, 256, eps=1e-6)]),
                    upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)))]),
                )),

                # 256 -> 128
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(256, 128, eps=1e-6), ResnetBlock2D(128, 128, eps=1e-6), ResnetBlock2D(128, 128, eps=1e-6)]),
                )),
            ]),

            # 128 -> 3
            conv_norm_out=nn.GroupNorm(32, 128, eps=1e-06),
            conv_act=nn.SiLU(),
            conv_out=nn.Conv2d(128, 3, kernel_size=3, padding=1)
        ))

        # fmt: on

    def encode(self, x, generator=None):
        h = x

        h = self.encoder["conv_in"](h)

        for down_block in self.encoder["down_blocks"]:
            for resnet in down_block["resnets"]:
                h = resnet(h)

            if "downsamplers" in down_block:
                h = F.pad(h, pad=(0, 1, 0, 1), mode="constant", value=0)
                h = down_block["downsamplers"][0]["conv"](h)

        h = self.encoder["mid_block"]["resnets"][0](h)
        h = self.encoder["mid_block"]["attentions"][0](h)
        h = self.encoder["mid_block"]["resnets"][1](h)

        h = self.encoder["conv_norm_out"](h)
        h = self.encoder["conv_act"](h)
        h = self.encoder["conv_out"](h)

        mean, logvar = self.quant_conv(h).chunk(2, dim=1)

        logvar = torch.clamp(logvar, -30.0, 20.0)

        std = torch.exp(0.5 * logvar)

        z = mean + torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator) * std

        z = z * 0.13025

        return z

    def decode(self, z):
        z = z / 0.13025

        h = z

        h = self.post_quant_conv(h)

        h = self.decoder["conv_in"](h)

        h = self.decoder["mid_block"]["resnets"][0](h)
        h = self.decoder["mid_block"]["attentions"][0](h)
        h = self.decoder["mid_block"]["resnets"][1](h)

        for up_block in self.decoder["up_blocks"]:
            for resnet in up_block["resnets"]:
                h = resnet(h)

            if "upsamplers" in up_block:
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                h = up_block["upsamplers"][0]["conv"](h)

        h = self.decoder["conv_norm_out"](h)
        h = self.decoder["conv_act"](h)
        h = self.decoder["conv_out"](h)

        x_pred = h

        return x_pred

    @classmethod
    def load_fp16_fix(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        if has_safetensors:
            load_from = "diffusion_pytorch_model.safetensors"
        else:
            load_from = "diffusion_pytorch_model.bin"

        return cls.load(hf_hub_download("madebyollin/sdxl-vae-fp16-fix", load_from), device=device)

    @classmethod
    def input_pil_to_tensor(self, x, include_batch_dim=True):
        x = np.array(x)
        x = torch.from_numpy(x).permute(2, 0, 1).to(torch.float32) / 255.0
        x = (x - 0.5) / 0.5
        if include_batch_dim and x.ndim == 3:
            x = x[None, :, :, :]
        return x

    @classmethod
    def output_tensor_to_pil(self, x_pred):
        from PIL import Image

        x_pred = ((x_pred * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        x_pred = [Image.fromarray(x) for x in x_pred]

        return x_pred


class SDXLUNet(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_outputs: Optional[List[torch.Tensor]] = None,
    ):
        if down_block_additional_residuals is not None:
            down_block_additional_residuals = list(down_block_additional_residuals)

        if add_to_down_block_outputs is not None:
            add_to_down_block_outputs = list(add_to_down_block_outputs)

        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = self.conv_in(hidden_state)

        residuals = [hidden_state]

        for down_block_idx, down_block in enumerate(self.down_blocks):
            for resnet_idx, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

                residuals.append(hidden_state)

            if down_block_idx != 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                residuals.append(hidden_state)

            if down_block_idx == 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        if mid_block_additional_residual is not None:
            hidden_state = hidden_state + mid_block_additional_residual

        for up_block in self.up_blocks:
            for resnet_idx, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)

        return hidden_state

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.safetensors"), device=device)

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.fp16.safetensors"), device=device)


class SDXLControlNet(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # controlnet cond embedding:
        self.controlnet_cond_embedding = nn.ModuleDict(dict(
            conv_in=nn.Conv2d(3, 16, kernel_size=3, padding=1),
            blocks=nn.ModuleList([
                # 16 -> 32
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
                # 32 -> 96
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 96, kernel_size=3, padding=1, stride=2),
                # 96 -> 256
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.Conv2d(96, 256, kernel_size=3, padding=1, stride=2),
            ]),
            conv_out=zero_module(nn.Conv2d(256, 320, kernel_size=3, padding=1)),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.controlnet_down_blocks = nn.ModuleList([
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.controlnet_mid_block = zero_module(nn.Conv2d(1280, 1280, kernel_size=1))

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        controlnet_cond,
    ):
        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        controlnet_cond = self.controlnet_cond_embedding["conv_in"](controlnet_cond)
        controlnet_cond = F.silu(controlnet_cond)

        for block in self.controlnet_cond_embedding["blocks"]:
            controlnet_cond = F.silu(block(controlnet_cond))

        controlnet_cond = self.controlnet_cond_embedding["conv_out"](controlnet_cond)

        hidden_state = self.conv_in(hidden_state)

        hidden_state = hidden_state + controlnet_cond

        down_block_res_sample = self.controlnet_down_blocks[0](hidden_state)
        down_block_res_samples = [down_block_res_sample]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][i](hidden_state, encoder_hidden_states)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        mid_block_res_sample = self.controlnet_mid_block(hidden_state)

        return dict(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

    @classmethod
    def from_unet(cls, unet):
        controlnet = cls()

        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet


class SDXLAdapter(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        self.adapter = nn.ModuleDict(dict(
            # 3 -> 768
            unshuffle=nn.PixelUnshuffle(16),

            # 768 -> 320
            conv_in=nn.Conv2d(768, 320, kernel_size=3, padding=1),

            body=nn.ModuleList([
                # 320 -> 320
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([
                        nn.ModuleDict(dict(block1=nn.Conv2d(320, 320, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(320, 320, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(320, 320, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(320, 320, kernel_size=1))),
                    ])
                )),
                # 320 -> 640
                nn.ModuleDict(dict(
                    in_conv=nn.Conv2d(320, 640, kernel_size=1),
                    resnets=nn.ModuleList([
                        nn.ModuleDict(dict(block1=nn.Conv2d(640, 640, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(640, 640, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(640, 640, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(640, 640, kernel_size=1))),
                    ])
                )),
                # 640 -> 1280
                nn.ModuleDict(dict(
                    downsample=nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    in_conv=nn.Conv2d(640, 1280, kernel_size=1),
                    resnets=nn.ModuleList([
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                    ])
                )),
                # 1280 -> 1280
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                    ])
                )),
            ])
        ))

        # fmt: on

    def forward(self, x):
        x = self.adapter["unshuffle"](x)
        x = self.adapter["conv_in"](x)

        features = []

        for block in self.adapter["body"]:
            if "downsample" in block:
                x = block["downsample"](x)

            if "in_conv" in block:
                x = block["in_conv"](x)

            for resnet in block["resnets"]:
                residual = x
                x = resnet["block1"](x)
                x = resnet["act"](x)
                x = resnet["block2"](x)
                x = residual + x

            features.append(x)

        return features


def get_sinusoidal_embedding(
    indices: torch.Tensor,
    embedding_dim: int,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=indices.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = indices.unsqueeze(-1).float() * emb
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    return emb


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim=None, eps=1e-5):
        super().__init__()

        if time_embedding_dim is not None:
            self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)
        else:
            self.time_emb_proj = None

        self.norm1 = torch.nn.GroupNorm(32, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels, eps=eps)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.nonlinearity = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, hidden_states, temb=None):
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            assert temb is not None
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual

        return hidden_states


class TransformerDecoder2D(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim, num_transformer_blocks):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels, eps=1e-06)
        self.proj_in = nn.Linear(channels, channels)

        self.transformer_blocks = nn.ModuleList([TransformerDecoderBlock(channels, encoder_hidden_states_dim) for _ in range(num_transformer_blocks)])

        self.proj_out = nn.Linear(channels, channels)

    def forward(self, hidden_states, encoder_hidden_states):
        batch_size, channels, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

        hidden_states = hidden_states + residual

        return hidden_states


class TransformerDecoderBlock(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = Attention(channels, channels)

        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = Attention(channels, encoder_hidden_states_dim)

        self.norm3 = nn.LayerNorm(channels)
        self.ff = nn.ModuleDict(dict(net=nn.Sequential(GEGLU(channels, 4 * channels), nn.Dropout(0.0), nn.Linear(4 * channels, channels))))

    def forward(self, hidden_states, encoder_hidden_states):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states

        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states

        hidden_states = self.ff["net"](self.norm3(hidden_states)) + hidden_states

        return hidden_states


_attention_implementation: Literal["xformers", "torch_2.0_scaled_dot_product"] = "torch_2.0_scaled_dot_product"


def set_attention_implementation(impl: Literal["xformers", "torch_2.0_scaled_dot_product"]):
    global _attention_implementation
    _attention_implementation = impl


def attention(to_q, to_k, to_v, to_out, head_dim, hidden_states, encoder_hidden_states=None, is_causal=False):
    batch_size, q_seq_len, channels = hidden_states.shape

    if encoder_hidden_states is not None:
        kv = encoder_hidden_states
    else:
        kv = hidden_states

    kv_seq_len = kv.shape[1]

    query = to_q(hidden_states)
    key = to_k(kv)
    value = to_v(kv)

    if _attention_implementation == "xformers":
        import xformers.ops

        query = query.reshape(batch_size, q_seq_len, channels // head_dim, head_dim).contiguous()
        key = key.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).contiguous()
        value = value.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).contiguous()

        if is_causal:
            attn_bias = xops.LowerTriangularMask()
        else:
            attn_bias = None

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attn_bias)

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.reshape(batch_size, q_seq_len, channels).contiguous()
    elif _attention_implementation == "torch_2.0_scaled_dot_product":
        query = query.reshape(batch_size, q_seq_len, channels // head_dim, head_dim).transpose(1, 2).contiguous()
        key = key.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).transpose(1, 2).contiguous()
        value = value.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).transpose(1, 2).contiguous()

        hidden_states = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, q_seq_len, channels).contiguous()
    else:
        assert False

    hidden_states = to_out(hidden_states)

    return hidden_states


class Attention(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim):
        super().__init__()
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(encoder_hidden_states_dim, channels, bias=False)
        self.to_v = nn.Linear(encoder_hidden_states_dim, channels, bias=False)
        self.to_out = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0.0))

    def forward(self, hidden_states, encoder_hidden_states=None):
        return attention(self.to_q, self.to_k, self.to_v, self.to_out, 64, hidden_states, encoder_hidden_states)


class VaeMidBlockAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-06)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0.0))
        self.head_dim = channels

    def forward(self, hidden_states):
        residual = hidden_states

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = attention(self.to_q, self.to_k, self.to_v, self.to_out, self.head_dim, hidden_states)

        hidden_states = hidden_states.transpose(1, 2).view(batch_size, channels, height, width)

        hidden_states = hidden_states + residual

        return hidden_states


class CLIPAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, hidden_states):
        return attention(self.q_proj, self.k_proj, self.v_proj, self.out_proj, 64, hidden_states, is_causal=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


def quick_gelu(input):
    return input * torch.sigmoid(1.702 * input)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# experimental models


class SDXLUNetControlnetPassThroughConditioning(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        self.conv_norm_out_1 = nn.GroupNorm(1, 9)
        self.conv_act_1 = nn.SiLU()
        self.conv_out_1 = nn.Conv2d(9, 4, kernel_size=3, padding=1)

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_outputs: Optional[List[torch.Tensor]] = None,
        controlnet_pass_through_conditioning=None,
    ):
        if down_block_additional_residuals is not None:
            down_block_additional_residuals = list(down_block_additional_residuals)

        if add_to_down_block_outputs is not None:
            add_to_down_block_outputs = list(add_to_down_block_outputs)

        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = self.conv_in(hidden_state)

        residuals = [hidden_state]

        for down_block_idx, down_block in enumerate(self.down_blocks):
            for resnet_idx, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

                residuals.append(hidden_state)

            if down_block_idx != 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                residuals.append(hidden_state)

            if down_block_idx == 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        if mid_block_additional_residual is not None:
            hidden_state = hidden_state + mid_block_additional_residual

        for up_block in self.up_blocks:
            for resnet_idx, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        residual = hidden_state

        hidden_state = torch.concat([hidden_state, controlnet_pass_through_conditioning], dim=1)

        hidden_state = self.conv_norm_out_1(hidden_state)
        hidden_state = self.conv_act_1(hidden_state)
        hidden_state = self.conv_out_1(hidden_state)

        return hidden_state

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.safetensors"), device=device, strict=False)

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.fp16.safetensors"), device=device, strict=False)


class SDXLUNetControlnetPassThroughConditioning_start_same_output(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        self.conv_norm_out_1 = nn.GroupNorm(1, 9)
        self.conv_act_1 = nn.SiLU()
        self.conv_out_1 = nn.Conv2d(9, 4, kernel_size=3, padding=1)
        zero_module(self.conv_norm_out_1)

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_outputs: Optional[List[torch.Tensor]] = None,
        controlnet_pass_through_conditioning=None,
    ):
        if down_block_additional_residuals is not None:
            down_block_additional_residuals = list(down_block_additional_residuals)

        if add_to_down_block_outputs is not None:
            add_to_down_block_outputs = list(add_to_down_block_outputs)

        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = self.conv_in(hidden_state)

        residuals = [hidden_state]

        for down_block_idx, down_block in enumerate(self.down_blocks):
            for resnet_idx, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

                residuals.append(hidden_state)

            if down_block_idx != 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                residuals.append(hidden_state)

            if down_block_idx == 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        if mid_block_additional_residual is not None:
            hidden_state = hidden_state + mid_block_additional_residual

        for up_block in self.up_blocks:
            for resnet_idx, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        residual = hidden_state

        hidden_state = torch.concat([hidden_state, controlnet_pass_through_conditioning], dim=1)

        hidden_state = self.conv_norm_out_1(hidden_state)
        hidden_state = self.conv_act_1(hidden_state)
        hidden_state = self.conv_out_1(hidden_state)
        hidden_state = hidden_state + residual

        return hidden_state

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.safetensors"), device=device, strict=False)

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.fp16.safetensors"), device=device, strict=False)


class SDXLControlNetPreEncodedControlnetCond(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        # unet latents: 4 + 
        # control image latents: 4 + 
        # controlnet_mask: 1
        # = 9 channels
        self.conv_in = nn.Conv2d(9, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.controlnet_down_blocks = nn.ModuleList([
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.controlnet_mid_block = zero_module(nn.Conv2d(1280, 1280, kernel_size=1))

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        controlnet_cond,
    ):
        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = torch.concat((hidden_state, controlnet_cond), dim=1)

        hidden_state = self.conv_in(hidden_state)

        down_block_res_sample = self.controlnet_down_blocks[0](hidden_state)
        down_block_res_samples = [down_block_res_sample]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][i](hidden_state, encoder_hidden_states)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        mid_block_res_sample = self.controlnet_mid_block(hidden_state)

        return dict(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

    @classmethod
    def from_unet(cls, unet):
        controlnet = cls()

        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        conv_in_weight = unet.conv_in.state_dict()["weight"]
        padding = torch.zeros((320, 5, 3, 3), device=conv_in_weight.device, dtype=conv_in_weight.dtype)
        conv_in_weight = torch.concat((conv_in_weight, padding), dim=1)

        conv_in_bias = unet.conv_in.state_dict()["bias"]

        controlnet.conv_in.load_state_dict({"weight": conv_in_weight, "bias": conv_in_bias})

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet


class SDXLUNetInpainting(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(9, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_outputs: Optional[List[torch.Tensor]] = None,
    ):
        if down_block_additional_residuals is not None:
            down_block_additional_residuals = list(down_block_additional_residuals)

        if add_to_down_block_outputs is not None:
            add_to_down_block_outputs = list(add_to_down_block_outputs)

        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = self.conv_in(hidden_state)

        residuals = [hidden_state]

        for down_block_idx, down_block in enumerate(self.down_blocks):
            for resnet_idx, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

                residuals.append(hidden_state)

            if down_block_idx != 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                residuals.append(hidden_state)

            if down_block_idx == 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        if mid_block_additional_residual is not None:
            hidden_state = hidden_state + mid_block_additional_residual

        for up_block in self.up_blocks:
            for resnet_idx, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][resnet_idx](hidden_state, encoder_hidden_states)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)

        return hidden_state

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        load_from = hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.safetensors")

        if version.parse(torch.__version__) >= version.parse("2.1"):
            state_dict = load_file(load_from, device=device)

            additional_weight = torch.zeros((320, 5, 3, 3), dtype=state_dict["conv_in.weight"].dtype, device=state_dict["conv_in.weight"].device)
            state_dict["conv_in.weight"] = torch.concat([state_dict["conv_in.weight"], additional_weight], dim=1)

            with torch.device("meta"):
                model = cls()

            model.load_state_dict(state_dict, assign=True)
        else:
            state_dict = load_file(load_from, device="cpu")

            additional_weight = torch.zeros((320, 5, 3, 3), dtype=state_dict["conv_in.weight"].dtype, device=state_dict["conv_in.weight"].device)
            state_dict["conv_in.weight"] = torch.concat([state_dict["conv_in.weight"], additional_weight], dim=1)

            with torch.device(device):
                model = cls()

            model.load_state_dict(state_dict)

            # HACK: this assumes all are the same dtype
            model.to(next(iter(state_dict.values())).dtype)

        # TODO - update conv_in
        # state_dict["conv_in.weight"] = torch.concat(state_dict["conv_in.weight"], torch.zeros())

        return model

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.fp16.safetensors"), device=device)


class SDXLUNetInpaintingCrossAttentionConditioning(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        self.conditioning_image_proj = nn.Linear(4, encoder_hidden_states_dim)
        self.conditioning_image_mask_embeddings = nn.Embedding(2, encoder_hidden_states_dim)

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    TransformerDecoder2D(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    TransformerDecoder2D(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        conditioning_image,
        conditioning_image_mask,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_outputs: Optional[List[torch.Tensor]] = None,
    ):
        if down_block_additional_residuals is not None:
            down_block_additional_residuals = list(down_block_additional_residuals)

        if add_to_down_block_outputs is not None:
            add_to_down_block_outputs = list(add_to_down_block_outputs)

        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        conditioning_image = self.conditioning_image_proj(conditioning_image.view(conditioning_image.shape[0], conditioning_image.shape[1], -1).permute(0, 2, 1))
        conditioning_image_mask = self.conditioning_image_mask_embeddings(conditioning_image_mask.long().view(conditioning_image_mask.shape[0], -1))
        cross_attention = torch.concat([encoder_hidden_states, conditioning_image + conditioning_image_mask], dim=1)

        hidden_state = self.conv_in(hidden_state)

        residuals = [hidden_state]

        for down_block_idx, down_block in enumerate(self.down_blocks):
            for resnet_idx, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][resnet_idx](hidden_state, cross_attention)

                residuals.append(hidden_state)

            if down_block_idx != 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                residuals.append(hidden_state)

            if down_block_idx == 0 and add_to_down_block_outputs is not None:
                hidden_state = hidden_state + add_to_down_block_outputs.pop(0)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, cross_attention)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        if mid_block_additional_residual is not None:
            hidden_state = hidden_state + mid_block_additional_residual

        for up_block in self.up_blocks:
            for resnet_idx, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][resnet_idx](hidden_state, cross_attention)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)

        return hidden_state

    @classmethod
    def load_fp32(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.safetensors"), device=device, strict=False)

    @classmethod
    def load_fp16(cls, device="cpu"):
        from huggingface_hub import hf_hub_download

        # TODO - shouldn't be necessary
        if not has_safetensors:
            raise ValueError("loading sdxl unet from huggingface hub checkpoint requires safetensors")

        return cls.load(hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", "unet/diffusion_pytorch_model.fp16.safetensors"), device=device, strict=False)
