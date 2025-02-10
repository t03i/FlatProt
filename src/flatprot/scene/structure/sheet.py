# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .base import StructureSceneElement


class SheetElement(StructureSceneElement):
    """A beta sheet element visualization using a simple triangular arrow"""

    def __init__(
        self,
        canvas_coordinates: np.ndarray,
        metadata: dict,
    ):
        super().__init__(canvas_coordinates, metadata)

    def calculate_display_coordinates(self) -> np.ndarray:
        start_point = self.coordinates[0]
        end_point = self.coordinates[-1]

        direction = end_point - start_point
        length = np.linalg.norm(direction)

        if (
            np.isclose(length, 0)
            or len(self.coordinates) <= self.style.min_sheet_length
        ):
            return np.array([start_point, end_point])

        perp = np.array([-direction[1], direction[0]])
        ribbon_width = self.style.line_width * self.style.base_width_factor

        left_point = start_point + perp * (ribbon_width / 2)
        right_point = start_point - perp * (ribbon_width / 2)

        return np.array([left_point, right_point, end_point])

    def display_coordinates_at_position(self, position: int) -> np.ndarray:
        """Returns the display coordinates for a specific position along the sheet.

        Args:
            position: Integer index of the position along the sheet

        Returns:
            np.ndarray: Array of coordinates for the triangular arrow at the given position
        """
        if len(self.display_coordinates) == 2 and position <= 1:
            return self.display_coordinates[position]

        start_point = self.display_coordinates[0]
        end_point = self.display_coordinates[-1]

        direction = end_point - start_point
        segment_length = np.linalg.norm(direction) / len(self.__coordinates)

        return start_point + direction * position * segment_length
