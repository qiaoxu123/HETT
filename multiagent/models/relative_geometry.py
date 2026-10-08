"""Vectorized candidate-to-agent / referenced-anchor geometry for dense SBF.

The world-aligned CityNav map uses normalized x toward east and normalized
image y toward south. Pose yaw is in radians, measured CCW from east.
Only observation-time pose and instruction-referenced landmark centroids are
allowed here; ground-truth target coordinates are NEVER used.
"""
import torch


def dense_relative_geometry(
    positions: torch.Tensor,
    heading_sin_cos: torch.Tensor,
    referenced_centroids: torch.Tensor,
    referenced_valid: torch.Tensor,
    *,
    field_size: int,
    map_meters: float,
) -> torch.Tensor:
    """Return [B,13,H,W] relative descriptors for each heatmap hypothesis.

    Channels 0..6: UAV-relative east/north displacement (normalized by map
    meters), metric distance/map_meters, absolute sin/cos bearing, egocentric
    sin/cos bearing. Channels 7..11: the same five SBF rho features relative
    to the MEAN instruction-referenced landmark centroid. Channel 12: anchor
    validity. This aggregate anchor is not a replacement for a per-landmark
    SBF selector when instructions name multiple landmarks.
    """
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [B,2]")
    if heading_sin_cos.shape != positions.shape:
        raise ValueError("heading_sin_cos must have shape [B,2]")
    if referenced_centroids.shape != positions.shape:
        raise ValueError("referenced_centroids must have shape [B,2]")
    if referenced_valid.shape != positions.shape[:1]:
        raise ValueError("referenced_valid must have shape [B]")
    if field_size < 1 or map_meters <= 0:
        raise ValueError("field_size and map_meters must be positive")

    batch = positions.shape[0]
    device, dtype = positions.device, positions.dtype
    axis = (torch.arange(field_size, device=device, dtype=dtype) + 0.5) / field_size
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    x = grid_x.unsqueeze(0)
    y = grid_y.unsqueeze(0)

    # CityNav: normalized y increases south; metric/world y increases north.
    dx = x - positions[:, 0, None, None]
    dy = positions[:, 1, None, None] - y
    distance = torch.sqrt(dx.square() + dy.square())
    safe_distance = distance.clamp_min(1e-6)
    sin_bearing = dy / safe_distance
    cos_bearing = dx / safe_distance

    sin_yaw = heading_sin_cos[:, 0, None, None]
    cos_yaw = heading_sin_cos[:, 1, None, None]
    heading_norm = torch.sqrt(sin_yaw.square() + cos_yaw.square()).clamp_min(1e-6)
    sin_yaw, cos_yaw = sin_yaw / heading_norm, cos_yaw / heading_norm
    sin_relative = sin_bearing * cos_yaw - cos_bearing * sin_yaw
    cos_relative = cos_bearing * cos_yaw + sin_bearing * sin_yaw

    ref_dx = x - referenced_centroids[:, 0, None, None]
    ref_dy = referenced_centroids[:, 1, None, None] - y
    ref_distance = torch.sqrt(ref_dx.square() + ref_dy.square())
    ref_safe_distance = ref_distance.clamp_min(1e-6)
    valid = referenced_valid.to(device=device, dtype=dtype).view(batch, 1, 1)
    valid_map = valid.expand(batch, field_size, field_size)

    channels = (
        dx, dy, distance, sin_bearing, cos_bearing,
        sin_relative, cos_relative,
        ref_dx * valid, ref_dy * valid, ref_distance * valid,
        (ref_dy / ref_safe_distance) * valid,
        (ref_dx / ref_safe_distance) * valid,
        valid_map,
    )
    # dx, dy, distances are normalized by map_meters (world metric distances
    # would be: values * map_meters); thus this is scale-stable across cities.
    return torch.stack(channels, dim=1)
