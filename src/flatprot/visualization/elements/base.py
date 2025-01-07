# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional, Literal

import numpy as np
import drawsvg as draw
from pydantic import BaseModel, Field, ConfigDict
from pydantic_extra_types.color import Color


class VisualizationStyle(BaseModel):
    """Base class for styling visualization elements"""

    fill_color: Color | Literal["none"] = Field(
        default="None", description="Element fill color"
    )
    stroke_color: Color | Literal["none"] = Field(
        default=Color("#000000"), description="Element stroke color"
    )
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Opacity value between 0 and 1"
    )
    line_width: float = Field(default=5.0, gt=0, description="Width of lines")
    smoothing_window: int = Field(
        default=5, ge=1, description="Window size for smoothing"
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
    def _smooth_coordinates(self, coords: np.ndarray, window: int = 1) -> np.ndarray:
        """Apply segment-based averaging to coordinates while preserving endpoints"""
        if window <= 1 or len(coords) <= 2:
            return coords

        # Preserve start and end points
        smoothed = np.zeros_like(coords)
        smoothed[0] = coords[0]
        smoothed[-1] = coords[-1]

        # Average points in segments
        for i in range(0, len(coords) - 2, window):
            segment = coords[i : min(i + window, len(coords) - 1)]
            avg_point = np.mean(segment, axis=0)
            smoothed[i + window // 2] = avg_point

        # Interpolate any missing points
        mask = np.all(smoothed == 0, axis=1)
        if np.any(mask):
            indices = np.arange(len(coords))
            for dim in range(coords.shape[1]):
                smoothed[mask, dim] = np.interp(
                    indices[mask], indices[~mask], smoothed[~mask, dim]
                )

        return smoothed
