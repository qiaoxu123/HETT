"""Guardrails that keep identity and GT labels out of visual encoders."""
from collections.abc import Mapping

FORBIDDEN_ENCODER_FIELDS = frozenset({
    "target_id", "object_id", "object_ids", "target_position", "target_xy",
    "candidate_id", "candidate_index", "candidate_rank", "correct_answer",
    "episode_id", "scene_key", "map_name", "GT", "target_label",
})


def assert_visual_only_inputs(batch):
    """Raise if a model batch includes metadata rather than only image pixels."""
    if not isinstance(batch, Mapping):
        raise TypeError("processor batch must be a mapping")
    bad = sorted(FORBIDDEN_ENCODER_FIELDS.intersection(batch))
    if bad:
        raise ValueError(f"forbidden labels passed to visual encoder: {bad}")
    if not set(batch).issubset({"pixel_values", "pixel_mask"}):
        raise ValueError(f"unexpected nonvisual model input keys: {sorted(set(batch) - {'pixel_values', 'pixel_mask'})}")
    return True
