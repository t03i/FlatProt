# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

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

    stroke_width_factor: float = Field(
        default=1.0, gt=0, description="Thickness relative to line width"
    )


class CoilVisualization(VisualizationElement, SmoothingMixin):
    """A coil element visualization using a smooth curved line"""

    def __init__(
        self,
        coordinates: np.ndarray,
        style: Optional[CoilStyle] = None,
    ):
        super().__init__(coordinates, style)

    def render(self) -> draw.DrawingElement:
        """Renders a coil using straight lines between atom coordinates.

        Returns:
            draw.Element: A path element representing the coil
        """
        # Smooth the coordinates to reduce noise
        coords = self._smooth_coordinates(
            self.coordinates, window=self.style.smoothing_window
        )

        # Create path
        path = draw.Path(
            stroke=self.style.stroke_color,
            stroke_width=self.style.line_width * self.style.stroke_width_factor,
            fill=self.style.fill_color,
            class_="coil",
        )

        # Start path at first point
        path.M(coords[0][0], coords[0][1])

        # Draw lines to each subsequent point
        for point in coords[1:]:
            path.L(point[0], point[1])

        return path
