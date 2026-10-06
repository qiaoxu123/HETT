"""GPU rendering backend.

The geometry is identical to the NumPy path and is cross-checked against it; the
difference is that the projection and the z-buffer run on the device, so a frame
costs one upload plus a handful of kernels instead of a multi-million element
sort on the CPU.

Point data is held as float32 positions and uint8 colours -- converting RGB to
float for the round trip would triple its footprint for no gain, since it is
only ever gathered, never arithmetic.
"""

from __future__ import annotations

import numpy as np

from .pointcloud_renderer import Camera, StageTimer

# Depth comparisons happen in float32; at a few hundred metres that type
# resolves to roughly 3e-5, so this tolerance only absorbs representation noise
# and not real depth differences.
DEPTH_EPS = 1e-3


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class DeviceCache:
    """Hot storage for block point data on the device.

    Two modes, both measured rather than assumed:

    ``mode="map"``   upload each block once and keep it resident.  A 59M-point
                     block is 708 MB of float32 positions plus 177 MB of uint8
                     colours, and it removes per-frame upload entirely.
    ``mode="span"``  upload only the span a frame needs.  Cheaper in device
                     memory, but the transfer recurs on every frame.

    ``budget_bytes`` bounds the total; blocks are evicted least-recently-used.
    """

    def __init__(self, device: str = "cuda", mode: str = "map",
                 budget_bytes: int = 24 * 1024 ** 3):
        self.device = device
        self.mode = mode
        self.budget_bytes = budget_bytes
        self._blocks = {}
        self._order = []
        self.bytes_resident = 0
        self.peak_bytes_resident = 0

    def get_block(self, key, xyz_source, rgb_source):
        """Whole-block tensors, uploaded once under ``mode='map'``."""
        import torch
        if self.mode != "map":
            return None
        if key in self._blocks:
            self._order.remove(key)
            self._order.append(key)
            return self._blocks[key]
        xyz = torch.as_tensor(_contig(xyz_source, np.float32), device=self.device)
        rgb = torch.as_tensor(_contig(rgb_source, np.uint8), device=self.device)
        size = xyz.numel() * xyz.element_size() + rgb.numel() * rgb.element_size()
        while self._blocks and self.bytes_resident + size > self.budget_bytes:
            old = self._order.pop(0)
            evicted = self._blocks.pop(old)
            self.bytes_resident -= (evicted[0].numel() * evicted[0].element_size()
                                    + evicted[1].numel() * evicted[1].element_size())
        self._blocks[key] = (xyz, rgb)
        self._order.append(key)
        self.bytes_resident += size
        self.peak_bytes_resident = max(self.peak_bytes_resident, self.bytes_resident)
        return self._blocks[key]

    def get_span(self, xyz_source, rgb_source, lo, hi):
        import torch
        xyz = torch.as_tensor(_contig(np.asarray(xyz_source[lo:hi]).copy(), np.float32),
                              device=self.device)
        rgb = torch.as_tensor(_contig(np.asarray(rgb_source[lo:hi]).copy(), np.uint8),
                              device=self.device)
        return xyz, rgb

    def clear(self):
        self._blocks.clear()
        self._order.clear()
        self.bytes_resident = 0
        try:
            import torch
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass


def camera_tensors(camera: Camera, device):
    """Camera basis and scalars as device tensors."""
    import torch
    right, up, forward = camera.basis()
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    return {
        "position": t(camera.position),
        "right": t(right), "up": t(up), "forward": t(forward),
        "focal": float(camera.focal),
        "near": float(camera.near), "far": float(camera.far),
        "tan_h": float(np.tan(np.deg2rad(camera.hfov_deg) / 2.0) * 1.02),
        "tan_v": float(np.tan(np.deg2rad(camera.vfov_deg) / 2.0) * 1.02),
        "width": camera.width, "height": camera.height,
    }


def project_torch(xyz, cam, positions):
    """Mirror of :func:`pointcloud_renderer.project_points`, on the device."""
    import torch
    rel = xyz - cam["position"]
    z_cam = rel[:, 0] * cam["forward"][0] + rel[:, 1] * cam["forward"][1] \
        + rel[:, 2] * cam["forward"][2]
    x_cam = rel[:, 0] * cam["right"][0] + rel[:, 1] * cam["right"][1] \
        + rel[:, 2] * cam["right"][2]
    y_cam = rel[:, 0] * cam["up"][0] + rel[:, 1] * cam["up"][1] \
        + rel[:, 2] * cam["up"][2]
    metric2 = rel[:, 0] ** 2 + rel[:, 1] ** 2 + rel[:, 2] ** 2

    candidate = (
        (z_cam > cam["near"])
        & (metric2 <= cam["far"] * cam["far"])
        & (x_cam.abs() <= z_cam * cam["tan_h"])
        & (y_cam.abs() <= z_cam * cam["tan_v"])
    )
    x_cam = x_cam[candidate]
    y_cam = y_cam[candidate]
    z_cam = z_cam[candidate]
    metric = torch.sqrt(metric2[candidate])
    positions = positions[candidate]

    inv_z = 1.0 / torch.clamp(z_cam, min=1e-9)
    u = cam["width"] / 2.0 + cam["focal"] * x_cam * inv_z
    v = cam["height"] / 2.0 - cam["focal"] * y_cam * inv_z
    inside = (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return u[inside], v[inside], metric[inside], positions[inside]


def render_torch(xyz, rgb, camera: Camera, positions, device: str = "cuda",
                 background=(24, 26, 30)) -> dict:
    """Render pre-uploaded device tensors; returns host arrays."""
    import torch
    h, w = camera.height, camera.width
    n_pix = h * w
    cam = camera_tensors(camera, device)

    u, v, metric, pos = project_torch(xyz, cam, positions)
    pix = v.to(torch.int64) * w + u.to(torch.int64)

    depth_buf = torch.full((n_pix,), float("inf"), dtype=torch.float32, device=device)
    if pix.numel():
        depth_buf.scatter_reduce_(0, pix, metric, reduce="amin", include_self=True)

    winner = torch.full((n_pix,), -1, dtype=torch.int64, device=device)
    if pix.numel():
        at_min = metric <= depth_buf[pix] + DEPTH_EPS
        if at_min.any():
            winner.scatter_reduce_(0, pix[at_min], pos[at_min].to(torch.int64),
                                   reduce="amax", include_self=True)

    valid = winner >= 0
    out = torch.full((n_pix, 3), 0, dtype=torch.uint8, device=device)
    out[:] = torch.as_tensor(np.array(background, dtype=np.uint8), device=device)
    if valid.any():
        out[valid] = rgb[winner[valid]]

    depth = torch.where(valid, depth_buf, torch.full_like(depth_buf, float("nan")))

    return {
        "rgb": out.reshape(h, w, 3).cpu().numpy(),
        "depth": depth.reshape(h, w).cpu().numpy().astype(np.float32),
        "valid": valid.reshape(h, w).cpu().numpy(),
        "point_pos": winner.reshape(h, w).cpu().numpy(),
        "n_projected": int(pix.numel()),
        "gpu_peak_bytes": (int(torch.cuda.max_memory_allocated(device))
                           if device.startswith("cuda") else 0),
    }


def render_cloud_region_torch(cloud, grid, camera: Camera, cache: DeviceCache,
                              lod=None, pad: float = 2.0,
                              timer: StageTimer | None = None,
                              zbuffer: str = "torch") -> dict:
    """Device rendering of the camera's neighbourhood, using the same query."""
    import torch

    from .pointcloud_renderer import cell_frustum_filter

    timer = timer or StageTimer()
    reach = camera.far + pad
    z_lo, z_hi = grid.exact_bounds()[0][2], grid.exact_bounds()[1][2]
    keep_cell = cell_frustum_filter(camera, z_lo, z_hi)

    with timer.stage("T_query"):
        slots = grid.query_radius_positions(
            camera.position[0], camera.position[1], reach, lod, keep_cell=keep_cell)
    timer.count("N_query", len(slots))

    with timer.stage("T_load"):
        xyz_sorted, rgb_sorted = grid.sorted_arrays()
        if cache.mode == "map":
            # The block lives on the device; the per-frame selection is a device
            # gather of the queried slots.  Projecting the whole block and
            # pairing it with a shorter position array would both waste work and
            # mismatch shapes.
            key = str(grid.sorted_paths[0])
            xyz_block, rgb_block = cache.get_block(key, xyz_sorted, rgb_sorted)
            slot_dev = _to_device(slots, cache.device)
            xyz_dev = xyz_block[slot_dev]
            rgb_dev = rgb_block[slot_dev]
            pos_dev = torch.arange(slot_dev.numel(), dtype=torch.int64,
                                   device=cache.device)
            base = 0
        else:
            lo = int(slots[0]) if slots.size else 0
            hi = int(slots[-1]) + 1 if slots.size else 0
            xyz_dev, rgb_dev = cache.get_span(xyz_sorted, rgb_sorted, lo, hi)
            pos_dev = _to_device(slots - lo, cache.device)
            base = lo
    timer.count("N_loaded", len(slots))

    import torch
    with timer.stage("T_zbuffer"):
        result = render_torch(xyz_dev, rgb_dev, camera, pos_dev, device=cache.device)
        torch.cuda.synchronize() if cache.device.startswith("cuda") else None
    timer.count("N_projected", result["n_projected"])
    timer.count("N_visible_points", int(result["valid"].sum()))
    timer.count("N_unique_pixels", int(result["valid"].sum()))

    result["point_pos"] = result["point_pos"] + base
    result["stats"] = {
        "camera": camera.to_json(),
        "zbuffer_backend": zbuffer,
        "gpu_peak_bytes": result["gpu_peak_bytes"],
        "device_cache_bytes": cache.bytes_resident,
        **timer.summary(),
    }
    return result


def _to_device(array, device):
    import torch
    return torch.as_tensor(_contig(array, np.int64), device=device)


def _contig(array, dtype):
    """A C-contiguous array of ``dtype``.

    torch refuses arrays with negative strides, so a reversed or strided view
    has to be copied rather than handed straight over.
    """
    return np.ascontiguousarray(array, dtype=dtype)


def render_points_torch(xyz: np.ndarray, rgb: np.ndarray, camera: Camera,
                        device: str = "cpu", background=(24, 26, 30)) -> dict:
    """Convenience entry point for a bare point array, mirroring ``render``.

    Used by the cross-backend tests so they can drive the device path without a
    grid or a block on disk.
    """
    import torch
    xyz_t = torch.as_tensor(_contig(xyz, np.float32), device=device)
    rgb_t = torch.as_tensor(_contig(np.asarray(rgb).reshape(-1, 3), np.uint8),
                            device=device)
    pos = torch.arange(xyz_t.shape[0], dtype=torch.int64, device=device)
    out = render_torch(xyz_t, rgb_t, camera, pos, device=device,
                       background=background)
    out["point_id"] = out.pop("point_pos")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out
