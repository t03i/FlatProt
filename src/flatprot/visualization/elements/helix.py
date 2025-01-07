# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

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

    thickness_factor: float = Field(
        default=15.0, gt=0, description="Thickness of helix relative to line width"
    )
    cross_width_factor: float = Field(
        default=25.0,
        gt=0,
        description="Width of helix wavelength relative to line width",
    )
    amplitude_factor: float = Field(
        default=5.0,
        gt=0,
        description="Amplitude of helix relative to line width",
    )
    min_helix_length: int = Field(
        default=4, gt=0, description="Minimum length of the helix for rendering"
    )
    stroke_width_factor: float = Field(
        default=0.5, gt=0, description="Width of the stroke relative to line width"
    )


class HelixVisualization(VisualizationElement, SmoothingMixin):
    """A helical element visualization using a zigzag style"""

    def __init__(
        self,
        coordinates: np.ndarray,
        style: Optional[HelixStyle] = None,
    ):
        super().__init__(coordinates, style)

    def _get_zigzag_points(self, start, end, thickness, wavelength, amplitude):
        """Calculate points for a sharp zigzag helix pattern."""
        # Convert inputs to numpy arrays
        start = np.array(start)
        end = np.array(end)

        # Calculate direction vector and length
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:  # Handle zero-length case
            return [start.tolist()]

        # Normalize direction vector and get perpendicular
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])

        # Calculate number of complete zigzag cycles
        num_cycles = max(1, int(length / wavelength))

        # Generate zigzag points (only peaks and valleys)
        t = np.linspace(0, length, num_cycles * 2 + 1)

        # Alternate between +amplitude and -amplitude
        wave = amplitude * np.array([1 if i % 2 == 0 else -1 for i in range(len(t))])

        # Create base points along the path
        base_points = t[:, None] * direction + start

        # Apply zigzag pattern
        wave_points = base_points + (wave[:, None] * perpendicular)

        # Create ribbon effect by offsetting top and bottom
        half_thickness = thickness / 2
        top_points = wave_points + (perpendicular * half_thickness)
        bottom_points = wave_points - (perpendicular * half_thickness)

        # Combine points to form complete ribbon outline
        return np.concatenate((top_points, bottom_points[::-1]))

    def render(self) -> draw.DrawingElement:
        """Renders a zigzag helix representation."""
        coords = self.coordinates

        if len(coords) <= self.style.min_helix_length:
            return draw.Line(
                *coords[0],
                *coords[-1],
                stroke=self.style.stroke_color,
                stroke_width=self.style.line_width * self.style.stroke_width_factor,
                class_="helix",
            )

        thickness = self.style.line_width * self.style.thickness_factor
        wavelength = self.style.line_width * self.style.cross_width_factor
        amplitude = self.style.line_width * self.style.amplitude_factor

        points = self._get_zigzag_points(
            coords[0], coords[-1], thickness, wavelength, amplitude
        )

        # Create a single connected path
        path = draw.Path(
            stroke=self.style.stroke_color,
            stroke_width=self.style.line_width * self.style.stroke_width_factor,
            fill=self.style.fill_color,
            class_="helix",
        )

        # Start path at first point
        path.M(*points[0])

        # Draw lines to each subsequent point
        for point in points[1:]:
            path.L(*point)

        # Close the path
        path.Z()

        return path
