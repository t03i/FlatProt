# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

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

    base_width_factor: float = Field(
        default=20, gt=0, description="Width of base relative to line width"
    )
    stroke_color: Color = Field(
        default=Color("#000000"), description="Color of the stroke"
    )


class SheetVisualization(VisualizationElement, SmoothingMixin):
    """A beta sheet element visualization using a simple triangular arrow"""

    def render(self) -> draw.DrawingElement:
        """Renders a simple triangular arrow using atom coordinates.

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
        if np.isclose(length, 0):
            return draw.Path(class_="sheet")  # Return empty path if length is zero

        # Normalize direction vector
        direction = direction / length

        # Calculate perpendicular vector
        perp = np.array([-direction[1], direction[0]])

        # Calculate ribbon width
        ribbon_width = self.style.line_width * self.style.base_width_factor

        # Calculate points for triangular arrow
        left_point = start_point + perp * (ribbon_width / 2)
        right_point = start_point - perp * (ribbon_width / 2)

        # Create path
        path = draw.Path(
            stroke=self.style.stroke_color,
            stroke_width=self.style.line_width,
            fill=self.style.color,
            class_="sheet",
        )

        # Draw simple triangular arrow
        path.M(left_point[0], left_point[1])  # Start at left point
        path.L(end_point[0], end_point[1])  # To arrow tip
        path.L(right_point[0], right_point[1])  # To right point
        path.Z()  # Close the path

        return path
