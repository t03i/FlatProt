# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Optional, Literal, Protocol

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
    smoothing_factor: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Percentage of points to keep"
    )

    model_config = ConfigDict(
        title="Base Visualization Style",
        frozen=True,  # Makes instances immutable
    )


class VisualizationElement(Protocol):
    """Base class for all visualization elements"""

    def __init__(self, style: Optional[VisualizationStyle] = None):
        self.style = style or VisualizationStyle(color="#000000")

    @abstractmethod
    def render(self, coordinates: np.ndarray) -> draw.DrawingElement:
        """Returns the SVG element"""
        pass

    def get_visual_style(self) -> VisualizationStyle:
        return self.style


class SmoothingMixin:
    def _smooth_coordinates(
        self, coords: np.ndarray, reduction_factor: float = 0.2
    ) -> np.ndarray:
        """Reduce point complexity using uniform selection.

        Args:
            coords: Input coordinates of shape (N, 2)
            reduction_factor: Fraction of points to keep (0.0-1.0)

        Returns:
            Simplified coordinates array
        """
        n_points = len(coords)
        if n_points <= 3 or reduction_factor >= 1.0:
            return coords

        # Always keep first and last points
        target_points = max(3, int(n_points * reduction_factor))
        if target_points >= n_points:
            return coords

        # Use linear indexing for uniform point selection
        indices = np.linspace(0, n_points - 1, target_points, dtype=int)
        return coords[indices]
