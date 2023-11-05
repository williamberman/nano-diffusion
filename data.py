import io
import json
import random
from typing import Tuple

import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as TF
from braceexpand import braceexpand
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, default_collate

from models import SDXLVae


def wds_dataloader_controlnet_inpainting_sdxl_synthetic_dataset(training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer):
    import webdataset as wds

    if isinstance(training_config.train_shards, list):
        train_shards = []
        for x in training_config.train_shards:
            train_shards.extend(braceexpand(x))
    elif isinstance(training_config.train_shards, str):
        train_shards = braceexpand(training_config.train_shards)
    else:
        assert False

    dataset = (
        wds.WebDataset(train_shards, resampled=True, handler=wds.warn_and_continue)
        .shuffle(training_config.shuffle_buffer_size)
        .map(lambda d: make_sample_controlnet_inpainting_sdxl_synthetic_dataset(d, training_config, tokenizer_one, tokenizer_two))
        .select(lambda sample: sample is not None)
        .batched(training_config.batch_size, partial=False, collation_fn=default_collate)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    return dataloader


@torch.no_grad()
def make_sample_controlnet_inpainting_sdxl_synthetic_dataset(sample, training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer):
    image = sdxl_synthetic_dataset_get_use_largest_clip_score(sample)

    if image is None:
        return None

    original_height = 1024
    original_width = 1024

    with io.BytesIO(image) as stream:
        image = PIL.Image.open(stream)
        image.load()
        image = image.convert("RGB")

    if random.random() < training_config.proportion_empty_prompts:
        text = ""
    else:
        text = sample["txt"].decode("utf-8")

    c_top, c_left, _, _ = get_random_crop_params([image.height, image.width], [1024, 1024])

    micro_conditioning = torch.tensor([original_width, original_height, c_top, c_left, 1024, 1024])

    text_input_ids_one = torch.tensor(tokenizer_one.encode(text).ids, dtype=torch.long)
    text_input_ids_two = torch.tensor(tokenizer_two.encode(text).ids, dtype=torch.long)

    image = TF.resize(
        image,
        1024,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )

    image = TF.crop(
        image,
        c_top,
        c_left,
        1024,
        1024,
    )

    conditioning_image = get_controlnet_inpainting_conditioning_image(image)
    conditioning_image = conditioning_image["conditioning_image"]

    return dict(
        micro_conditioning=micro_conditioning,
        text_input_ids_one=text_input_ids_one,
        text_input_ids_two=text_input_ids_two,
        image=SDXLVae.input_pil_to_tensor(image, include_batch_dim=False),
        conditioning_image=conditioning_image,
    )


def sdxl_synthetic_dataset_get_use_largest_clip_score(sample):
    if "clip_scores.txt" not in sample:
        return None

    clip_scores = sample["clip_scores.txt"].decode("utf-8")
    clip_scores = clip_scores.split(",")
    clip_scores = [float(x) for x in clip_scores]

    index_of_max = 0

    for i in range(1, len(clip_scores)):
        if clip_scores[i] > clip_scores[index_of_max]:
            index_of_max = i

    key_of_best_clip_score_image = f"{index_of_max}.png"

    if key_of_best_clip_score_image not in sample:
        raise ValueError(
            f"{key_of_best_clip_score_image} was not found in sample. The dataset should have files <sample key>.<x>.png where <x> coresponds to an index of the clip scores in clip_scores.txt"
        )

    return sample[key_of_best_clip_score_image]


def wds_dataloader_controlnet_inpainting(training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer):
    import webdataset as wds

    if isinstance(training_config.train_shards, list):
        train_shards = []
        for x in training_config.train_shards:
            train_shards.extend(braceexpand(x))
    elif isinstance(training_config.train_shards, str):
        train_shards = braceexpand(training_config.train_shards)
    else:
        assert False

    dataset = (
        wds.WebDataset(train_shards, resampled=True, handler=wds.warn_and_continue)
        .shuffle(training_config.shuffle_buffer_size)
        .map(lambda d: make_sample_controlnet_inpainting(d, training_config, tokenizer_one, tokenizer_two))
        .select(lambda sample: sample is not None)
        .batched(training_config.batch_size, partial=False, collation_fn=default_collate)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    return dataloader


@torch.no_grad()
def make_sample_controlnet_inpainting(sample, training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer):
    image = sample["png"]
    metadata = json.loads(sample["json"].decode("utf-8"))
    original_width = int(metadata.get("original_width", 0.0))
    original_height = int(metadata.get("original_height", 0.0))

    with io.BytesIO(image) as stream:
        image = PIL.Image.open(stream)
        image.load()
        image = image.convert("RGB")

    if random.random() < training_config.proportion_empty_prompts:
        text = ""
    else:
        text = sample["txt"].decode("utf-8")

    c_top, c_left, _, _ = get_random_crop_params([image.height, image.width], [1024, 1024])

    micro_conditioning = torch.tensor([original_width, original_height, c_top, c_left, 1024, 1024])

    text_input_ids_one = torch.tensor(tokenizer_one.encode(text).ids, dtype=torch.long)
    text_input_ids_two = torch.tensor(tokenizer_two.encode(text).ids, dtype=torch.long)

    image = TF.resize(
        image,
        1024,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )

    image = TF.crop(
        image,
        c_top,
        c_left,
        1024,
        1024,
    )

    conditioning_image = get_controlnet_inpainting_conditioning_image(image)
    conditioning_image = conditioning_image["conditioning_image"]

    return dict(
        micro_conditioning=micro_conditioning,
        text_input_ids_one=text_input_ids_one,
        text_input_ids_two=text_input_ids_two,
        image=SDXLVae.input_pil_to_tensor(image, include_batch_dim=False),
        conditioning_image=conditioning_image,
    )


def wds_dataloader_unet_inpainting_hq_dataset(training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer, return_dataloader=True):
    import webdataset as wds

    if isinstance(training_config.train_shards, list):
        train_shards = []
        for x in training_config.train_shards:
            train_shards.extend(braceexpand(x))
    elif isinstance(training_config.train_shards, str):
        train_shards = braceexpand(training_config.train_shards)
    else:
        assert False

    dataset = (
        wds.WebDataset(train_shards, resampled=True, handler=wds.warn_and_continue)
        .shuffle(training_config.shuffle_buffer_size)
        .map(lambda d: make_sample_unet_inpainting_hq_dataset(d, training_config, tokenizer_one, tokenizer_two))
        .select(lambda sample: sample is not None)
        .batched(training_config.batch_size, partial=False, collation_fn=default_collate)
    )

    if return_dataloader:
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )
    else:
        return dataset


@torch.no_grad()
def make_sample_unet_inpainting_hq_dataset(sample, training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer):
    image = sample["png"]

    with io.BytesIO(image) as stream:
        image = PIL.Image.open(stream)
        image.load()
        image = image.convert("RGB")

    if random.random() < training_config.proportion_empty_prompts:
        text = ""
    else:
        text = sample["txt"].decode("utf-8")

    text_input_ids_one = torch.tensor(tokenizer_one.encode(text).ids, dtype=torch.long)
    text_input_ids_two = torch.tensor(tokenizer_two.encode(text).ids, dtype=torch.long)

    original_height = image.height
    original_width = image.width

    image = TF.resize(
        image,
        1024,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )

    c_top, c_left, _, _ = get_random_crop_params([image.height, image.width], [1024, 1024])

    micro_conditioning = torch.tensor([original_height, original_width, c_top, c_left, 1024, 1024])

    image = TF.crop(
        image,
        c_top,
        c_left,
        1024,
        1024,
    )

    conditioning_image = get_unet_inpainting_conditioning_image(image)

    return dict(
        micro_conditioning=micro_conditioning,
        text_input_ids_one=text_input_ids_one,
        text_input_ids_two=text_input_ids_two,
        image=SDXLVae.input_pil_to_tensor(image, include_batch_dim=False),
        conditioning_image=conditioning_image["conditioning_image"],
        conditioning_image_mask=conditioning_image["conditioning_image_mask"],
    )


def wds_dataloader_unet_text_to_image_hq_dataset(training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer, return_dataloader=True):
    import webdataset as wds

    if isinstance(training_config.train_shards, list):
        train_shards = []
        for x in training_config.train_shards:
            train_shards.extend(braceexpand(x))
    elif isinstance(training_config.train_shards, str):
        train_shards = braceexpand(training_config.train_shards)
    else:
        assert False

    dataset = (
        wds.WebDataset(train_shards, resampled=True, handler=wds.warn_and_continue)
        .shuffle(training_config.shuffle_buffer_size)
        .map(lambda d: make_sample_unet_text_to_image_hq_dataset(d, training_config, tokenizer_one, tokenizer_two))
        .select(lambda sample: sample is not None)
        .batched(training_config.batch_size, partial=False, collation_fn=default_collate)
    )

    if return_dataloader:
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )
    else:
        return dataset


@torch.no_grad()
def make_sample_unet_text_to_image_hq_dataset(sample, training_config, tokenizer_one: Tokenizer, tokenizer_two: Tokenizer):
    image = sample["png"]

    with io.BytesIO(image) as stream:
        image = PIL.Image.open(stream)
        image.load()
        image = image.convert("RGB")

    if random.random() < training_config.proportion_empty_prompts:
        text = ""
    else:
        text = sample["txt"].decode("utf-8")

    text_input_ids_one = torch.tensor(tokenizer_one.encode(text).ids, dtype=torch.long)
    text_input_ids_two = torch.tensor(tokenizer_two.encode(text).ids, dtype=torch.long)

    original_height = image.height
    original_width = image.width

    image = TF.resize(
        image,
        1024,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )

    c_top, c_left, _, _ = get_random_crop_params([image.height, image.width], [1024, 1024])

    micro_conditioning = torch.tensor([original_height, original_width, c_top, c_left, 1024, 1024])

    image = TF.crop(
        image,
        c_top,
        c_left,
        1024,
        1024,
    )

    return dict(
        micro_conditioning=micro_conditioning,
        text_input_ids_one=text_input_ids_one,
        text_input_ids_two=text_input_ids_two,
        image=SDXLVae.input_pil_to_tensor(image, include_batch_dim=False),
    )


def get_random_crop_params(input_size: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = input_size

    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()

    return i, j, th, tw


def get_controlnet_canny_conditioning_image(image):
    import cv2

    conditioning_image = np.array(image)
    conditioning_image = cv2.Canny(conditioning_image, 100, 200)
    conditioning_image = conditioning_image[:, :, None]
    conditioning_image = np.concatenate([conditioning_image, conditioning_image, conditioning_image], axis=2)

    conditioning_image_as_pil = Image.fromarray(conditioning_image)

    conditioning_image = conditioning_image.permute(2, 0, 1).to(torch.float32) / 255.0

    return dict(conditioning_image=conditioning_image, conditioning_image_as_pil=conditioning_image_as_pil)


def get_controlnet_inpainting_conditioning_image(image):
    conditioning_image_mask = make_random_mask(image.height, image.width)

    conditioning_image = torch.from_numpy(np.array(image))
    conditioning_image = conditioning_image.permute(2, 0, 1).to(torch.float32) / 255.0
    unmasked_conditioning_image = conditioning_image

    # Just zero out the pixels which will be masked
    conditioning_image_as_pil = conditioning_image * (conditioning_image_mask < 0.5)
    conditioning_image_as_pil = (conditioning_image_as_pil.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    conditioning_image_as_pil = Image.fromarray(conditioning_image_as_pil)

    # where mask is set to 1, set to -1 "special" masked image pixel.
    # -1 is outside of the 0-1 range that the controlnet normalized
    # input is in.
    conditioning_image = conditioning_image * (conditioning_image_mask < 0.5) + -1.0 * (conditioning_image_mask >= 0.5)

    return dict(
        conditioning_image=conditioning_image,
        conditioning_image_as_pil=conditioning_image_as_pil,
        conditioning_image_mask=conditioning_image_mask,
        unmasked_conditioning_image=unmasked_conditioning_image,
    )


def get_unet_inpainting_conditioning_image(image):
    conditioning_image_mask = make_random_mask(image.height, image.width)

    conditioning_image = torch.from_numpy(np.array(image))
    conditioning_image = conditioning_image.permute(2, 0, 1).to(torch.float32) / 255.0
    unmasked_conditioning_image = conditioning_image

    # Just zero out the pixels which will be masked
    conditioning_image_as_pil = conditioning_image * (conditioning_image_mask < 0.5)
    conditioning_image_as_pil = (conditioning_image_as_pil.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    conditioning_image_as_pil = Image.fromarray(conditioning_image_as_pil)

    conditioning_image = conditioning_image * (conditioning_image_mask < 0.5)
    conditioning_image = (conditioning_image - 0.5) / 0.5

    return dict(
        conditioning_image=conditioning_image,
        conditioning_image_as_pil=conditioning_image_as_pil,
        conditioning_image_mask=conditioning_image_mask,
        unmasked_conditioning_image=unmasked_conditioning_image,
    )


def make_random_mask(height, width):
    if random.random() <= 0.25:
        mask = np.ones((height, width), np.float32)
    else:
        mask = random.choice([make_random_rectangle_mask, make_random_irregular_mask, make_outpainting_mask])(height, width)

    mask = torch.from_numpy(mask)

    mask = mask[None, :, :]

    return mask


def make_random_rectangle_mask(
    height,
    width,
    margin=10,
    bbox_min_size=100,
    bbox_max_size=512,
    min_times=1,
    max_times=2,
):
    mask = np.zeros((height, width), np.float32)

    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)

        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)

        mask[start_y : start_y + box_height, start_x : start_x + box_width] = 1

    return mask


def make_random_irregular_mask(height, width, max_angle=4, max_len=60, max_width=256, min_times=1, max_times=2):
    import cv2

    mask = np.zeros((height, width), np.float32)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)

        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)

            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle

            length = 10 + np.random.randint(max_len)

            brush_w = 5 + np.random.randint(max_width)

            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)

            choice = random.randint(0, 2)

            if choice == 0:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif choice == 1:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1.0, thickness=-1)
            elif choice == 2:
                radius = brush_w // 2
                mask[
                    start_y - radius : start_y + radius,
                    start_x - radius : start_x + radius,
                ] = 1
            else:
                assert False

            start_x, start_y = end_x, end_y

    return mask


def make_outpainting_mask(height, width, probs=[0.5, 0.5, 0.5, 0.5]):
    mask = np.zeros((height, width), np.float32)
    at_least_one_mask_applied = False

    coords = [
        [(0, 0), (1, get_padding(height))],
        [(0, 0), (get_padding(width), 1)],
        [(0, 1 - get_padding(height)), (1, 1)],
        [(1 - get_padding(width), 0), (1, 1)],
    ]

    for pp, coord in zip(probs, coords):
        if np.random.random() < pp:
            at_least_one_mask_applied = True
            mask = apply_padding(mask=mask, coord=coord)

    if not at_least_one_mask_applied:
        idx = np.random.choice(range(len(coords)), p=np.array(probs) / sum(probs))
        mask = apply_padding(mask=mask, coord=coords[idx])

    return mask


def get_padding(size, min_padding_percent=0.04, max_padding_percent=0.5):
    n1 = int(min_padding_percent * size)
    n2 = int(max_padding_percent * size)
    return np.random.randint(n1, n2) / size


def apply_padding(mask, coord):
    height, width = mask.shape

    mask[
        int(coord[0][0] * height) : int(coord[1][0] * height),
        int(coord[0][1] * width) : int(coord[1][1] * width),
    ] = 1

    return mask
