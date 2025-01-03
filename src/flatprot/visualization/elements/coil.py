# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from pydantic import Field
import numpy as np
import drawsvg as draw

from flatprot.visualization.elements import (
    VisualizationElement,
    VisualizationStyle,
    SmoothingMixin,
)


class CoilStyle(VisualizationStyle):
    """Settings specific to coil visualization"""

    line_thickness_factor: float = Field(
        default=0.5, gt=0, description="Thickness relative to line width"
    )
    spline_points: int = Field(
        default=20, ge=2, description="Number of points for spline interpolation"
    )


@dataclass
class CoilVisualization(VisualizationElement, SmoothingMixin):
    """A coil element visualization using a smooth curved line"""

    def __init__(
        self,
        coordinates: np.ndarray,
        style: Optional[CoilStyle] = None,
    ):
        super().__init__(coordinates, style)

    def _interpolate_points(self, coords: np.ndarray) -> np.ndarray:
        """Create a smooth interpolation between points"""
        num_points = self.style.spline_points

        # If we have very few points, just return the original coordinates
        if len(coords) < 3:
            return coords

        # Generate parameter values for interpolation
        t = np.linspace(0, 1, len(coords))
        t_new = np.linspace(0, 1, num_points)

        # Interpolate x and y coordinates separately
        x_interpolated = np.interp(t_new, t, coords[:, 0])
        y_interpolated = np.interp(t_new, t, coords[:, 1])

        return np.column_stack((x_interpolated, y_interpolated))

    def render(self) -> draw.DrawingElement:
        """Renders a smooth coil using atom coordinates.

        Returns:
            draw.Element: A path element representing the coil
        """
        # Smooth the coordinates to reduce noise
        coords = self._smooth_coordinates(self.coordinates)

        # Interpolate additional points for smoother curve
        points = self._interpolate_points(coords)

        # Create path
        path = draw.Path(
            stroke=self.style.color,
            stroke_width=self.style.line_width * self.style.line_thickness_factor,
            fill="none",
            class_="coil",
        )

        # Start path
        path.M(points[0][0], points[0][1])

        # Add smooth curve through points
        for i in range(1, len(points) - 2, 2):
            # Calculate control points for smooth curve
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            x3, y3 = points[min(i + 2, len(points) - 1)]

            # Use quadratic Bezier curve for smooth transition
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            path.Q(x2, y2, x3, y3)

        # Add final segment if needed
        if len(points) > 2:
            path.L(points[-1][0], points[-1][1])

        return path
