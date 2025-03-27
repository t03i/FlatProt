# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from .base import StructureSceneElement


def calculate_zigzag_points(start, end, thickness, wavelength, amplitude):
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
    wave = amplitude * np.array([1 if i % 2 == 0 else -1 for i in range(len(t) - 2)])
    wave = np.concatenate(([0], wave, [0]))

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


class HelixElement(StructureSceneElement):
    """A helical element visualization using a zigzag style"""

    def calculate_display_coordinates(self) -> np.ndarray:
        """Calculate the display coordinates for the helix.

        For helices shorter than min_helix_length, returns just start and end coordinates.
        For normal helices, returns zigzag ribbon coordinates.

        Returns:
            np.ndarray: Array of coordinates forming the helix visualization
        """
        if len(self._coordinates) <= self.style.min_helix_length:
            return self._coordinates[[0, -1]]  # Just return start and end points

        return calculate_zigzag_points(
            self._coordinates[0],
            self._coordinates[-1],
            self.style.line_width * self.style.ribbon_thickness_factor,
            self.style.line_width * self.style.wavelength,
            self.style.line_width * self.style.amplitude,
        )

    def calculate_display_coordinates_at_resiude(self, residue_idx: int) -> np.ndarray:
        """Maps a position from the original coordinate system to the display coordinates.

        For short helices (length <= min_helix_length), interpolates linearly between
        start and end points. For normal helices, finds the corresponding point on the
        central zigzag line.

        Args:
            residue_idx: Integer index in the original coordinate system

        Returns:
            np.ndarray: Corresponding point in display coordinates

        Raises:
            IndexError: If no coordinates are available
        """
        if len(self._coordinates) == 0:
            raise IndexError("No coordinates available")

        if len(self._coordinates) <= 1:
            return self._coordinates[0]

        # For short helices, interpolate linearly
        if len(self._coordinates) <= self.style.min_helix_length:
            progress = residue_idx / (len(self._coordinates) - 1)
            return (1 - progress) * self._coordinates[0] + progress * self._coordinates[
                -1
            ]

        # Normal zigzag helix calculation
        display_coords = self._display_coordinates
        wave_points = len(display_coords) // 2
        progress = residue_idx / (len(self._coordinates) - 1)
        wave_position = min(int(progress * (wave_points - 1)), wave_points - 1)

        # Get points from top and bottom waves
        top_point = display_coords[wave_position]
        bottom_point = display_coords[-(wave_position + 1)]

        return (top_point + bottom_point) / 2
