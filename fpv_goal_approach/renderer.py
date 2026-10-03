"""Renderer backends.

`orthophoto_heightfield` is an explicit 2.5-D approximation for smoke tests. It
is not an original CityFlight frame. AirSim backends preserve the raw Pose5D
pitch and require a separately running CityNav Unreal environment.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import rasterio


class FPVRenderer(ABC):
    source: str

    @abstractmethod
    def render(self, map_name: str, pose: Sequence[float], output_path: Path) -> Path:
        raise NotImplementedError


class OrthoHeightfieldRenderer(FPVRenderer):
    """Fast 2.5-D perspective proxy from published orthophoto RGB and DSM.

    This backend retains camera yaw and pitch but cannot reconstruct facades or
    occlusions faithfully. Metadata always records `orthophoto_heightfield`.
    """

    source = "orthophoto_heightfield"

    def __init__(self, rgbd_dir: Path, image_size: int = 224, horizontal_fov_deg: float = 90.0):
        self.rgbd_dir = Path(rgbd_dir)
        self.image_size = int(image_size)
        self.horizontal_fov = math.radians(horizontal_fov_deg)
        self._cache: dict[str, tuple[np.ndarray, rasterio.DatasetReader]] = {}

    def _load(self, map_name: str):
        if map_name not in self._cache:
            rgb = cv2.imread(str(self.rgbd_dir / f"{map_name}.png"), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(self.rgbd_dir / f"{map_name}.png")
            self._cache[map_name] = (rgb, rasterio.open(self.rgbd_dir / f"{map_name}.tif"))
        return self._cache[map_name]

    def render(self, map_name: str, pose: Sequence[float], output_path: Path) -> Path:
        rgb, raster = self._load(map_name)
        x, y, z, yaw, pitch = map(float, pose)
        size = self.image_size
        # A perspective trapezoid in world XY. Pitch controls near/far reach;
        # this is deliberately named as a heightfield proxy, never CityFlight.
        downward = float(np.clip(-pitch / (math.pi / 2), 0.05, 1.0))
        far_m = 25.0 + (1.0 - downward) * 110.0
        near_m = max(2.0, z * 0.04)
        half_near = near_m * math.tan(self.horizontal_fov / 2)
        half_far = far_m * math.tan(self.horizontal_fov / 2)
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        origin = np.array([x, y])
        corners = np.stack([
            origin + forward * near_m - right * half_near,
            origin + forward * near_m + right * half_near,
            origin + forward * far_m + right * half_far,
            origin + forward * far_m - right * half_far,
        ])
        src = np.array([[raster.index(px, py)[1], raster.index(px, py)[0]] for px, py in corners], dtype=np.float32)
        dst = np.array([[0, size - 1], [size - 1, size - 1], [size - 1, 0], [0, 0]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(src, dst)
        frame = cv2.warpPerspective(
            rgb, transform, (size, size), borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127),
        )
        # Darken the horizon slightly so out-of-bounds pixels are explicit.
        frame[: max(1, size // 20)] = (frame[: max(1, size // 20)] * 0.7).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise OSError(f"failed to write {output_path}")
        return output_path


class AirSimRenderer(FPVRenderer):
    def __init__(self, host: str, port: int, perspective: str, image_size: int = 224):
        if perspective not in {"front", "slanted"}:
            raise ValueError(perspective)
        import airsim

        self.airsim = airsim
        self.client = airsim.VehicleClient(ip=host, port=port)
        self.client.confirmConnection()
        self.perspective = perspective
        self.source = f"airsim_{perspective}"
        self.image_size = image_size
        self.map_name = None

    def render(self, map_name: str, pose: Sequence[float], output_path: Path) -> Path:
        airsim = self.airsim
        if self.map_name != map_name:
            self.client.simLoadLevel(map_name[0] + map_name.split("_")[-1])
            self.map_name = map_name
        x, y, z, yaw, pitch = map(float, pose)
        camera_pitch = pitch + (-math.pi / 4 if self.perspective == "slanted" else 0.0)
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(x, -y, -z), airsim.to_quaternion(camera_pitch, 0, -yaw)),
            ignore_collision=True,
        )
        response = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ])[0]
        if not response.image_data_uint8:
            raise RuntimeError(f"AirSim returned no image for {map_name}")
        frame = np.frombuffer(response.image_data_uint8, np.uint8).reshape(response.height, response.width, 3)
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        return output_path


def frame_key(map_name: str, pose: Sequence[float], source: str) -> str:
    payload = f"{source}|{map_name}|" + "|".join(f"{float(value):.5f}" for value in pose)
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def create_renderer(source: str, rgbd_dir: Path, image_size: int, host: str, port: int) -> FPVRenderer:
    if source == "orthophoto_heightfield":
        return OrthoHeightfieldRenderer(rgbd_dir, image_size=image_size)
    if source in {"airsim_front", "airsim_slanted"}:
        return AirSimRenderer(host, port, source.removeprefix("airsim_"), image_size=image_size)
    raise ValueError(f"unsupported render source: {source}")
