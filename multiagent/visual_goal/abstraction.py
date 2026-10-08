"""Interpretable goal-template transformations and controlled ablations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualMasks:
    target: np.ndarray
    anchor: np.ndarray
    building: np.ndarray
    road: np.ndarray
    parking: np.ndarray
    vegetation: np.ndarray
    open_area: np.ndarray


def _mask(mask, shape):
    if mask is None:
        return np.zeros(shape[:2], dtype=bool)
    return np.asarray(mask).astype(bool)


def _blur(image, sigma):
    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def _keep_regions(image, keep, background=(112, 112, 112)):
    keep = _mask(keep, image.shape)
    out = np.empty_like(image)
    out[:] = background
    out[keep] = image[keep]
    return out


def _posterize(image, colors):
    # Per-channel uniform quantization is deterministic and does not fit on
    # validation data or use an image-specific clustering objective.
    bins = max(2, min(32, int(colors)))
    step = 256.0 / bins
    return np.clip((np.floor(image.astype(np.float32) / step) + 0.5) * step, 0, 255).astype(np.uint8)


def _semantic_image(image, masks: VisualMasks):
    h, w = image.shape[:2]
    out = np.full((h, w, 3), (190, 190, 190), dtype=np.uint8)
    palette = (
        ("building", masks.building, (170, 170, 170)),
        ("road", masks.road, (80, 80, 80)),
        ("parking", masks.parking, (130, 130, 130)),
        ("vegetation", masks.vegetation, (100, 160, 80)),
        ("open", masks.open_area, (215, 215, 215)),
    )
    for _, mask, color in palette:
        out[_mask(mask, image.shape)] = color
    out[_mask(masks.anchor, image.shape)] = (40, 115, 245)
    out[_mask(masks.target, image.shape)] = (245, 70, 55)
    return out


def _contour_image(image, masks: VisualMasks, color=True, include_context=True):
    h, w = image.shape[:2]
    out = np.full((h, w, 3), (25, 25, 25), dtype=np.uint8)
    if color:
        coarse = cv2.resize(cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(out, 0.72, coarse, 0.28, 0)
    layers = [
        (masks.building, (155, 155, 155)),
        (masks.road, (105, 105, 105)),
        (masks.parking, (125, 125, 125)),
        (masks.vegetation, (115, 115, 115)),
        (masks.anchor, (40, 125, 255)),
        (masks.target, (255, 70, 60)),
    ]
    for layer_index, (mask, color_value) in enumerate(layers):
        if not include_context and layer_index < 4:
            continue
        u8 = (_mask(mask, image.shape).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color_value, 1, lineType=cv2.LINE_8)
    return out


def make_levels(image: np.ndarray, masks: VisualMasks, poster_colors=8) -> dict[str, np.ndarray]:
    """Return L0-L9; query images are intentionally never transformed here."""
    image = np.asarray(image, dtype=np.uint8)
    union = _mask(masks.target, image.shape) | _mask(masks.anchor, image.shape)
    low = cv2.resize(cv2.resize(image, (12, 12), interpolation=cv2.INTER_AREA),
                     (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.repeat(gray[..., None], 3, axis=2)
    semantic = _semantic_image(image, masks)
    context_contour = _contour_image(image, masks, color=True, include_context=True)
    pure_contour = _contour_image(image, masks, color=False, include_context=True)
    # L9 encodes only target and anchor size/layout. Its mask rendering is
    # category-coded; no RGB pixel from the source appears in it.
    geometry = np.full_like(image, 245)
    geometry[_mask(masks.anchor, image.shape)] = (80, 80, 80)
    geometry[_mask(masks.target, image.shape)] = (30, 30, 30)
    return {
        "L0": image.copy(),
        "L1": np.where(union[..., None], image, _blur(image, 12)),
        "L2": _keep_regions(image, union),
        "L3": low,
        "L4": _posterize(image, poster_colors),
        "L5": gray,
        "L6": semantic,
        "L7": context_contour,
        "L8": pure_contour,
        "L9": geometry,
    }


def controlled_ablation(image: np.ndarray, masks: VisualMasks) -> dict[str, np.ndarray]:
    """Single-factor / set-composition ablations for target-anchor-context."""
    image = np.asarray(image, dtype=np.uint8)
    target = _mask(masks.target, image.shape)
    anchor = _mask(masks.anchor, image.shape)
    context = (_mask(masks.building, image.shape) | _mask(masks.road, image.shape)
               | _mask(masks.parking, image.shape) | _mask(masks.vegetation, image.shape)
               | _mask(masks.open_area, image.shape))
    no_texture = cv2.resize(cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA),
                            (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 70, 150) > 0
    return {
        "color_full": image.copy(),
        "color_gray": np.repeat(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None], 3, 2),
        "color_4": _posterize(image, 4),
        "color_8": _posterize(image, 8),
        "color_16": _posterize(image, 16),
        "color_coarse": _posterize(image, 8),
        "color_none": np.repeat(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None], 3, 2),
        "texture_full": image.copy(),
        "texture_blur": _blur(image, 4),
        "texture_strong_blur": _blur(image, 12),
        "texture_none": no_texture,
        "geometry_full": image.copy(),
        "geometry_contour": np.repeat(edges[..., None] * 255, 3, 2).astype(np.uint8),
        "geometry_coarse_footprint": _keep_regions(image, cv2.dilate((target | anchor).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0),
        "geometry_none": np.full_like(image, 128),
        "context_no_road": _keep_regions(image, target | anchor | (context & ~_mask(masks.road, image.shape))),
        "context_no_parking": _keep_regions(image, target | anchor | (context & ~_mask(masks.parking, image.shape))),
        "context_no_buildings": _keep_regions(image, target | anchor | (context & ~_mask(masks.building, image.shape))),
        "context_no_vegetation": _keep_regions(image, target | anchor | (context & ~_mask(masks.vegetation, image.shape))),
        "context_no_open_area": _keep_regions(image, target | anchor | (context & ~_mask(masks.open_area, image.shape))),
        "target_only": _keep_regions(image, target),
        "anchor_only": _keep_regions(image, anchor),
        "target_anchor": _keep_regions(image, target | anchor),
        "target_anchor_context": _keep_regions(image, target | anchor | context),
    }
