"""Hand-crafted image features and a small linear/MLP probe."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def handcrafted_features(rgb):
    rgb = np.asarray(rgb, np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    values = []
    for image, ranges in ((hsv, ((0, 180), (0, 256), (0, 256))), (lab, ((0, 256),) * 3)):
        for channel, value_range in enumerate(ranges):
            hist = cv2.calcHist([image], [channel], None, [16], list(value_range)).flatten(); hist /= max(hist.sum(), 1)
            values.extend(hist)
    for image in (rgb, hsv, lab):
        values.extend(image.reshape(-1, 3).mean(0) / 255); values.extend(image.reshape(-1, 3).std(0) / 255)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    values += [edges.mean() / 255, gray.mean() / 255, gray.std() / 255]
    return np.asarray(values, np.float32)


class AttributeProbe(nn.Module):
    def __init__(self, input_dim, classes, hidden_dim=0):
        super().__init__()
        self.network = nn.Linear(input_dim, classes) if not hidden_dim else nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, classes))

    def forward(self, values): return self.network(values)

