"""SensatUrban + CityNav first-person RGB-D rendering, without Unreal or AirSim.

The question this package answers: can a CityNav ``Pose4D`` be mapped into the
SensatUrban point cloud's coordinate system, and does a perspective projection of
the raw coloured points then form a geometrically consistent first-person view
that landmark grounding could be built on?

Reading order::

    plyio                    -- block access and the XY bucket index
    citynav                  -- episodes, poses, CityRefer landmarks
    coordinate_diagnostic    -- is it the same frame?  (bounds / z / landmarks)
    fit_coordinate_transform -- search the small transform hypothesis class
    pointcloud_renderer      -- pinhole projection with a z-buffer
    project_landmarks        -- landmark projection and visibility
    render_trajectory_fpv    -- per-pose artifacts
    validate_rendering       -- the six gates
"""

from __future__ import annotations

__all__ = [
    "plyio",
    "citynav",
    "coordinate_diagnostic",
    "fit_coordinate_transform",
    "pointcloud_renderer",
    "project_landmarks",
    "render_trajectory_fpv",
    "validate_rendering",
]
