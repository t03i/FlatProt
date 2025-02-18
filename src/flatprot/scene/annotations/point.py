# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .base import AnnotationElement


class PointAnnotation(AnnotationElement):
    """A point annotation that marks specific positions in the scene."""

    def display_coordinates(self) -> np.ndarray:
        """Calculate the coordinates of the point annotation.

        Returns:
            np.ndarray: Array of coordinates for each point in the annotation
        """
        # Get coordinates for each position from its corresponding element
        return np.array(
            [
                element.display_coordinates_at_position(pos)
                for element, pos in zip(self._elements, self._positions)
            ]
        )
