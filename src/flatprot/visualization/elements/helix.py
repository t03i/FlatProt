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


class HelixStyle(VisualizationStyle):
    """Settings specific to helix visualization"""

    wave_height_factor: float = Field(
        default=0.75, gt=0, description="Height of helix waves relative to line width"
    )
    wave_frequency: float = Field(
        default=2.0, gt=0, description="Number of waves per segment"
    )
    min_waves: int = Field(
        default=2, ge=1, description="Minimum number of waves to render"
    )
    ribbon_thickness_factor: float = Field(
        default=0.5, gt=0, description="Thickness of ribbon relative to line width"
    )
    stroke_color: Color = Field(
        default=Color("#000000"), description="Color of the stroke"
    )


class HelixVisualization(VisualizationElement, SmoothingMixin):
    """A helical element visualization using a ribbon style"""

    def __init__(
        self,
        coordinates: np.ndarray,
        style: Optional[HelixStyle] = None,
    ):
        super().__init__(coordinates, style)

    def render(self) -> draw.DrawingElement:
        """Renders a ribbon-style helix using all atom coordinates.

        Returns:
            draw.Element: A path element representing the ribbon helix
        """
        # Smooth the coordinates to reduce noise
        coords = self._smooth_coordinates(self.coordinates)

        # Calculate path length for wave scaling
        total_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))

        # Calculate number of waves based on length
        num_waves = max(
            self.style.min_waves,
            int(total_length / (self.style.line_width * self.style.wave_frequency)),
        )

        # Parameters
        wave_height = self.style.line_width * self.style.wave_height_factor
        thickness = self.style.line_width * self.style.ribbon_thickness_factor

        # Generate points along the helix path
        num_points = num_waves * 2 + 1
        t_values = np.linspace(0, 1, num_points)

        # Interpolate positions along the coordinate path
        indices = t_values * (len(coords) - 1)
        int_indices = indices.astype(int)
        fractions = indices - int_indices

        points = (
            coords[int_indices] * (1 - fractions[:, np.newaxis])
            + coords[np.minimum(int_indices + 1, len(coords) - 1)]
            * fractions[:, np.newaxis]
        )

        # Calculate direction vectors along the path
        directions = np.diff(points, axis=0)
        directions = np.vstack([directions[0], directions])  # Pad first direction
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        # Calculate perpendicular vectors
        perp_vectors = np.stack([-directions[:, 1], directions[:, 0]], axis=1)

        # Generate wave offsets
        wave_offsets = wave_height * np.sin(
            np.linspace(0, num_waves * 2 * np.pi, num_points)
        )

        # Calculate ribbon points
        wave_points = points + perp_vectors * wave_offsets[:, np.newaxis]
        upper_points = wave_points + perp_vectors * thickness
        lower_points = wave_points - perp_vectors * thickness

        # Create path
        path = draw.Path(
            stroke=self.style.stroke_color,
            stroke_width=1,
            fill=self.style.color,
            class_="helix",
        )

        # Add upper edge
        path.M(upper_points[0][0], upper_points[0][1])
        for point in upper_points[1:]:
            path.L(point[0], point[1])

        # Add lower edge in reverse
        for point in lower_points[::-1]:
            path.L(point[0], point[1])

        # Close the path
        path.Z()

        return path
