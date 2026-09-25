"""Small semantic detector used only for the projection feasibility gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSES = ("Background", "Building", "Car", "Ground", "Parking")


class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.SiLU())

    def forward(self, value):
        return self.layers(value)


class TinyUNet(nn.Module):
    def __init__(self, classes=len(CLASSES), width=24):
        super().__init__()
        self.enc1 = Block(3, width)
        self.enc2 = Block(width, width * 2)
        self.enc3 = Block(width * 2, width * 4)
        self.middle = Block(width * 4, width * 8)
        self.dec3 = Block(width * 8 + width * 4, width * 4)
        self.dec2 = Block(width * 4 + width * 2, width * 2)
        self.dec1 = Block(width * 2 + width, width)
        self.output = nn.Conv2d(width, classes, 1)

    def forward(self, value):
        first = self.enc1(value)
        second = self.enc2(F.max_pool2d(first, 2))
        third = self.enc3(F.max_pool2d(second, 2))
        value = self.middle(F.max_pool2d(third, 2))
        value = self.dec3(torch.cat((F.interpolate(value, scale_factor=2), third), 1))
        value = self.dec2(torch.cat((F.interpolate(value, scale_factor=2), second), 1))
        value = self.dec1(torch.cat((F.interpolate(value, scale_factor=2), first), 1))
        return self.output(value)
