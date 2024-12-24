# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import drawsvg as draw
from pydantic import BaseModel, Field, ConfigDict
from pydantic_extra_types.color import Color


class VisualizationStyle(BaseModel):
    """Base class for styling visualization elements"""

    color: Color = Field(default=Color("#000000"), description="Element color")
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Opacity value between 0 and 1"
    )
    line_width: float = Field(default=1.0, gt=0, description="Width of lines")
    simplification: float = Field(
        default=0.0, ge=0.0, description="Simplification factor"
    )
    size: float = Field(default=1.0, gt=0, description="Size multiplier")
    smoothing_window: int = Field(
        default=3, ge=1, description="Window size for smoothing"
    )

    model_config = ConfigDict(
        title="Base Visualization Style",
        frozen=True,  # Makes instances immutable
    )


class VisualizationElement(ABC):
    """Base class for all visualization elements"""

    def __init__(
        self, coordinates: np.ndarray, style: Optional[VisualizationStyle] = None
    ):
        self.style = style or VisualizationStyle(color="#000000")
        self.coordinates = coordinates

    @abstractmethod
    def render(self) -> draw.DrawingElement:
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
