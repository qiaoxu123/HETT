from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np


def _mask(image: np.ndarray, regions: Mapping[str, np.ndarray] | None, names: tuple[str, ...]) -> np.ndarray:
    h, w = image.shape[:2]
    out = np.zeros((h, w), np.uint8)
    for name in names:
        if regions and name in regions:
            candidate = np.asarray(regions[name], dtype=np.uint8)
            if candidate.shape != (h, w):
                candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_NEAREST)
            out |= (candidate > 0).astype(np.uint8)
    return out.astype(bool)


def abstract(image: np.ndarray, level: str, regions: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """Return an interpretable L0-L9 template; input/output are uint8 RGB."""
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    h, w = rgb.shape[:2]
    if level == "L0":
        return rgb.copy()
    anchors = _mask(rgb, regions, ("target", "anchor"))
    target = _mask(rgb, regions, ("target",))
    anchor = _mask(rgb, regions, ("anchor",))
    context = _mask(rgb, regions, ("building", "road", "parking", "vegetation", "open_area"))
    if level == "L1":
        blur = cv2.GaussianBlur(rgb, (0, 0), max(3, min(h, w) / 12))
        keep = cv2.dilate(anchors.astype(np.uint8), np.ones((max(3, h // 30), max(3, w // 30)), np.uint8)) > 0
        return np.where(keep[..., None], rgb, blur)
    if level == "L2":
        return np.where(anchors[..., None], rgb, 127).astype(np.uint8)
    if level == "L3":
        small = cv2.resize(rgb, (max(2, w // 16), max(2, h // 16)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if level == "L4" or level in {"L4_4", "L4_8", "L4_16"}:
        # Deterministic color clustering with the requested palette size.
        colors = 8 if level == "L4" else int(level.split("_")[1])
        pixels = rgb.reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(pixels, colors, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        return centers[labels.ravel()].reshape(rgb.shape).clip(0, 255).astype(np.uint8)
    if level == "L5":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.repeat(gray[..., None], 3, axis=2)
    if level == "L6":
        out = np.full_like(rgb, (24, 24, 24))
        palette = {"building": (170, 70, 50), "road": (110, 110, 110), "parking": (190, 190, 170),
                   "vegetation": (40, 145, 55), "open_area": (210, 205, 160), "anchor": (40, 90, 220), "target": (240, 35, 35)}
        for name, color in palette.items():
            if regions and name in regions:
                out[np.asarray(regions[name]) > 0] = color
        return out
    if level == "L7":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        out = np.zeros_like(rgb)
        out[edges > 0] = (180, 180, 180)
        out[anchor] = (40, 100, 230)
        out[target] = (255, 30, 30)
        return out
    if level == "L8":
        geometry = _mask(rgb, regions, ("target", "anchor", "building", "road_skeleton"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150) > 0
        out = np.zeros_like(rgb)
        out[edges & geometry] = 255
        out[target] = (255, 255, 255)
        out[anchor] = (180, 180, 180)
        return out
    if level == "L9":
        out = np.zeros_like(rgb)
        # Encode target/anchor footprints without preserving texture or color.
        out[anchor] = (140, 140, 140)
        out[target] = (255, 255, 255)
        return out
    raise ValueError(f"unknown abstraction level: {level}")


def information_table() -> list[dict[str, str]]:
    return [
        {"level":"L0","texture":"yes","color":"yes","background":"yes","semantic":"implicit","geometry":"yes"},
        {"level":"L1","texture":"local","color":"yes","background":"blurred","semantic":"implicit","geometry":"yes"},
        {"level":"L2","texture":"anchors","color":"yes","background":"removed","semantic":"implicit","geometry":"anchor masks"},
        {"level":"L3","texture":"no","color":"coarse","background":"yes","semantic":"implicit","geometry":"coarse"},
        {"level":"L4","texture":"yes","color":"quantized","background":"yes","semantic":"implicit","geometry":"yes"},
        {"level":"L5","texture":"yes","color":"no","background":"yes","semantic":"implicit","geometry":"yes"},
        {"level":"L6","texture":"no","color":"class palette","background":"coarse","semantic":"yes","geometry":"yes"},
        {"level":"L7","texture":"no","color":"coarse","background":"removed","semantic":"partial","geometry":"contour"},
        {"level":"L8","texture":"no","color":"no","background":"removed","semantic":"no","geometry":"contour/skeleton"},
        {"level":"L9","texture":"no","color":"no","background":"removed","semantic":"target/anchor","geometry":"relative footprints"},
    ]
