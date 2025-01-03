# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import numpy as np
import drawsvg as draw
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.visualization.elements import (
    VisualizationElement,
    VisualizationStyle,
    SmoothingMixin,
)


class SheetStyle(VisualizationStyle):
    """Settings specific to sheet visualization"""

    ribbon_thickness_factor: float = Field(
        default=0.8, gt=0, description="Thickness of sheet relative to line width"
    )
    arrow_width_factor: float = Field(
        default=1.2, gt=0, description="Width of arrow head relative to ribbon"
    )
    arrow_length_factor: float = Field(
        default=2.0, gt=0, description="Length of arrow head relative to width"
    )
    stroke_color: Color = Field(
        default=Color("#000000"), description="Color of the stroke"
    )


class SheetVisualization(VisualizationElement, SmoothingMixin):
    """A beta sheet element visualization using a rectangular ribbon with arrow"""

    def __init__(
        self,
        coordinates: np.ndarray,
        style: Optional[SheetStyle] = None,
    ):
        super().__init__(coordinates, style)

    def render(self) -> draw.DrawingElement:
        """Renders a rectangular sheet with arrow using atom coordinates.

        Returns:
            draw.Element: A path element representing the sheet
        """
        # Smooth the coordinates to reduce noise
        coords = self._smooth_coordinates(self.coordinates)

        # Get start and end points
        start_point = coords[0]
        end_point = coords[-1]

        # Calculate direction vector
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        if length == 0:
            return draw.Path()  # Return empty path if length is zero

        direction = direction / length

        # Calculate perpendicular vector
        perp = np.array([-direction[1], direction[0]])

        # Calculate dimensions
        thickness = self.style.line_width * self.style.ribbon_thickness_factor
        arrow_width = thickness * self.style.arrow_width_factor
        arrow_length = thickness * self.style.arrow_length_factor

        # Calculate arrow base point (slightly before end_point)
        arrow_base = end_point - direction * arrow_length

        # Calculate corner points for rectangular part
        upper_start = start_point + perp * (thickness / 2)
        lower_start = start_point - perp * (thickness / 2)
        upper_arrow_base = arrow_base + perp * (thickness / 2)
        lower_arrow_base = arrow_base - perp * (thickness / 2)

        # Calculate arrow points
        arrow_upper = arrow_base + perp * arrow_width
        arrow_lower = arrow_base - perp * arrow_width

        # Create path
        path = draw.Path(
            stroke=self.style.stroke_color, stroke_width=1, fill=self.style.color
        )

        # Draw rectangular part and arrow
        path.M(upper_start[0], upper_start[1])  # Start at upper left
        path.L(upper_arrow_base[0], upper_arrow_base[1])  # To upper right before arrow
        path.L(arrow_upper[0], arrow_upper[1])  # To arrow upper point
        path.L(end_point[0], end_point[1])  # To arrow tip
        path.L(arrow_lower[0], arrow_lower[1])  # To arrow lower point
        path.L(lower_arrow_base[0], lower_arrow_base[1])  # To lower right before arrow
        path.L(lower_start[0], lower_start[1])  # To lower left
        path.Z()  # Close the path

        return path
