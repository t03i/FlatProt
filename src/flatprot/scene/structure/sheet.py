# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .base import StructureSceneElement


class SheetElement(StructureSceneElement):
    """A beta sheet element visualization using a simple triangular arrow"""

    def calculate_display_coordinates(self) -> np.ndarray:
        start_point = self._coordinates[0]
        end_point = self._coordinates[-1]

        direction = (end_point - start_point) / np.linalg.norm(end_point - start_point)
        length = np.linalg.norm(start_point - end_point)

        if (
            np.isclose(length, 0)
            or len(self._coordinates) <= self.style.min_sheet_length
        ):
            return np.array([start_point, end_point])

        perp = np.array([-direction[1], direction[0]])
        arrow_width = self.style.line_width * self.style.arrow_width_factor

        left_point = start_point + perp * (arrow_width / 2)
        right_point = start_point - perp * (arrow_width / 2)

        return np.array([left_point, right_point, end_point])

    def calculate_display_coordinates_at_resiude(self, residue_idx: int) -> np.ndarray:
        """Returns the display coordinates for a specific position along the sheet.

        Args:
            position: Integer index of the position along the sheet

        Returns:
            np.ndarray: Array of coordinates for the triangular arrow at the given position
        """
        if len(self._display_coordinates) == 2 and residue_idx <= 1:
            return self._display_coordinates[residue_idx]

        end_point = self._display_coordinates[-1]
        midpoint = (self._display_coordinates[0] + self._display_coordinates[1]) / 2

        direction = (end_point - midpoint) / np.linalg.norm(end_point - midpoint)
        segment_length = np.linalg.norm(end_point - midpoint) / len(self._coordinates)

        return midpoint + direction * residue_idx * segment_length
