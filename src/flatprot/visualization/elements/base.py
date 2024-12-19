# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from drawsvg import draw


@dataclass
class VisualizationStyle:
    """Base class for styling visualization elements"""

    color: str
    opacity: float = 1.0
    line_width: float = 1.0
    simplification: float = 0
    size: float = 1.0
    smoothing_window: int = 3


class VisualizationElement(ABC):
    """Base class for all visualization elements"""

    def __init__(
        self, coordinates: np.ndarray, style: Optional[VisualizationStyle] = None
    ):
        self.style = style or VisualizationStyle(color="#000000")
        self.coordinates = coordinates

    @abstractmethod
    def render(self) -> draw.Element:
        """Returns the SVG element"""
        pass


class SmoothingMixin:
    def _smooth_coordinates(self, coords: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing to coordinates"""
        if window <= 1:
            return coords

        # Pad the ends to maintain array size
        pad_width = window // 2
        padded = np.pad(coords, ((pad_width, pad_width), (0, 0)), mode="edge")

        # Calculate moving average
        smoothed = np.zeros_like(coords)
        for i in range(len(coords)):
            smoothed[i] = np.mean(padded[i : i + window], axis=0)

        return smoothed
