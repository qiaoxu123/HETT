"""Independent SensatUrban point-cloud perspective rendering utilities."""

from pathlib import Path

import numpy as np
import torch


PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("class", "u1"),
])

# LBN classes: Background, Building, Car, Ground, Parking. Other valid
# SensatUrban categories remain Background; missing pixels are masked separately.
SENSAT_TO_LBN = np.asarray([3, 0, 1, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0], dtype=np.uint8)


def camera_basis(direction):
    forward = np.asarray(direction, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-4:
        horizontal = forward.copy()
        horizontal[2] = 0.0
        if np.linalg.norm(horizontal) < 1e-4:
            horizontal = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(
            horizontal / np.linalg.norm(horizontal), np.asarray([0.0, 0.0, 1.0]))
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-9)
    return right, up, forward


def unproject_depth(depth, pose, hfov_deg=90.0):
    """Recover world XYZ from forward-axis depth and the renderer camera model."""
    height, width = depth.shape
    right, up, forward = camera_basis(pose[3:6])
    focal = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    vv, uu = np.indices((height, width), dtype=np.float64)
    x_ratio = (uu - width / 2.0) / focal
    y_ratio = (height / 2.0 - vv) / focal
    rays = (forward[None, None] + x_ratio[..., None] * right[None, None]
            + y_ratio[..., None] * up[None, None])
    return np.asarray(pose[:3], dtype=np.float64) + depth[..., None] * rays


def read_ply(path: Path, stride: int = 8):
    count = None
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {path}")
            if line.startswith(b"element vertex "):
                count = int(line.split()[2])
            if line.strip() == b"end_header":
                offset = stream.tell()
                break
    if count is None:
        raise ValueError(f"PLY has no vertex count: {path}")
    cloud = np.memmap(path, mode="r", dtype=PLY_DTYPE, offset=offset, shape=(count,))
    sampled = cloud[::stride]
    xyz = np.column_stack((sampled["x"], sampled["y"], sampled["z"])).astype(np.float32)
    rgb = np.column_stack((sampled["red"], sampled["green"], sampled["blue"])).astype(np.uint8)
    semantic = SENSAT_TO_LBN[np.minimum(sampled["class"], len(SENSAT_TO_LBN) - 1)]
    return xyz, rgb, semantic


class PerspectivePointRenderer:
    """Z-buffer point renderer that preserves validity and semantic labels."""

    def __init__(self, xyz, rgb, semantic, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.xyz = torch.as_tensor(xyz, device=self.device)
        self.rgb = torch.as_tensor(rgb, device=self.device)
        self.semantic = torch.as_tensor(semantic, device=self.device)

    @torch.inference_mode()
    def render(self, pose, width=128, height=128, hfov_deg=90.0, max_range=260.0):
        pose = np.asarray(pose, dtype=np.float64)
        camera = torch.as_tensor(pose[:3], dtype=torch.float32, device=self.device)
        right_np, up_np, forward_np = camera_basis(pose[3:6])
        forward = torch.as_tensor(forward_np, dtype=torch.float32, device=self.device)
        right = torch.as_tensor(right_np, dtype=torch.float32, device=self.device)
        up = torch.as_tensor(up_np, dtype=torch.float32, device=self.device)

        relative = self.xyz - camera
        local = relative.square().sum(-1).le(max_range * max_range)
        indices = local.nonzero(as_tuple=False).flatten()
        relative = relative[indices]
        depth = relative @ forward
        x_cam, y_cam = relative @ right, relative @ up
        front = depth.gt(0.5)
        indices, depth, x_cam, y_cam = (
            value[front] for value in (indices, depth, x_cam, y_cam))
        focal = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
        u = torch.round(width / 2.0 + focal * x_cam / depth).long()
        v = torch.round(height / 2.0 - focal * y_cam / depth).long()
        inside = u.ge(0) & u.lt(width) & v.ge(0) & v.lt(height)
        indices, depth, u, v = (value[inside] for value in (indices, depth, u, v))

        pixel = v * width + u
        min_depth = torch.full(
            (height * width,), float("inf"), dtype=torch.float32, device=self.device)
        min_depth.scatter_reduce_(0, pixel, depth, reduce="amin", include_self=True)
        nearest = depth.le(min_depth[pixel] + 1e-4)
        selected = torch.full(
            (height * width,), -1, dtype=torch.long, device=self.device)
        selected[pixel[nearest]] = indices[nearest]
        valid = selected.ge(0)

        image = torch.zeros((height * width, 3), dtype=torch.uint8, device=self.device)
        labels = torch.full((height * width,), 255, dtype=torch.uint8, device=self.device)
        depth_image = torch.full(
            (height * width,), float("nan"), dtype=torch.float32, device=self.device)
        image[valid] = self.rgb[selected[valid]]
        labels[valid] = self.semantic[selected[valid]]
        depth_image[valid] = min_depth[valid]
        return {
            "rgb": image.reshape(height, width, 3).cpu().numpy(),
            "semantic": labels.reshape(height, width).cpu().numpy(),
            "depth": depth_image.reshape(height, width).cpu().numpy(),
            "valid": valid.reshape(height, width).cpu().numpy(),
        }
