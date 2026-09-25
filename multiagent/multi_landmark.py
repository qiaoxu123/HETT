"""Target-free per-landmark feature lookup for HETT rollouts."""

from dataclasses import dataclass
from collections import defaultdict
import hashlib
from typing import Tuple

import torch


def episode_key(value) -> Tuple[str, str, int]:
    return str(value[0]), str(value[1]), int(value[2])


@dataclass(frozen=True)
class MultiLandmarkBatch:
    centers: torch.Tensor
    boundaries: torch.Tensor
    valid: torch.Tensor
    relation: torch.Tensor
    cue: torch.Tensor


class MultiLandmarkIndex:
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != 1:
            raise ValueError("unsupported multi-landmark artifact")
        self.max_landmarks = int(payload["max_landmarks"])
        self.rows = {}
        for split in payload["splits"].values():
            for index, episode_id in enumerate(split["episode_ids"]):
                key = episode_key(episode_id)
                if key in self.rows:
                    raise ValueError(f"duplicate episode id: {key}")
                self.rows[key] = (
                    split["centers"][index], split["boundaries"][index],
                    split["valid"][index], split["relation"][index],
                    split["cue"][index],
                )
        groups = defaultdict(list)
        for key, row in self.rows.items():
            groups[(key[0], int(row[2].sum()))].append(key)
        self.wrong_relation = self._different_switch(groups, (3, 4))
        self.wrong_landmark = self._different_switch(groups, (0, 1, 2))

    def _different_switch(self, groups, fields):
        """Choose a same-map/count control whose requested tensors differ."""
        switched = {}
        for values in groups.values():
            signatures = {}
            representatives = {}
            for key in values:
                digest = hashlib.blake2b(digest_size=16)
                for field in fields:
                    digest.update(self.rows[key][field].contiguous().numpy().tobytes())
                signature = digest.digest()
                signatures[key] = signature
                representatives.setdefault(signature, key)
            distinct = list(representatives)
            distinct_index = {signature: index
                              for index, signature in enumerate(distinct)}
            for key in values:
                own = signatures[key]
                if len(distinct) == 1:
                    switched[key] = key
                else:
                    other = distinct[(distinct_index[own] + 1) % len(distinct)]
                    switched[key] = representatives[other]
        return switched

    def batch(self, episode_ids, device, control="normal"):
        values = []
        missing = []
        for episode_id in episode_ids:
            key = episode_key(episode_id)
            row = self.rows.get(key)
            if row is None:
                missing.append(episode_key(episode_id))
            values.append(row)
        if missing:
            raise KeyError(f"missing multi-landmark rows: {missing[:3]}")
        if control in ("wrong_relation", "wrong_landmark"):
            mapping = (self.wrong_relation if control == "wrong_relation"
                       else self.wrong_landmark)
            switched = [self.rows[mapping[episode_key(value)]]
                        for value in episode_ids]
            if control == "wrong_relation":
                values = [row[:3] + other[3:] for row, other in zip(values, switched)]
            else:
                values = [other[:3] + row[3:] for row, other in zip(values, switched)]
        fields = list(zip(*values))
        return MultiLandmarkBatch(*[
            torch.stack(field).to(device) for field in fields
        ])
