# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .base import Annotation

class LineAnnotation(Annotation):
    def display_coordinates(self) -> np.ndarray:
        assert len(self.indices) == 2, "LineAnnotation must have exactly two indices"
        assert 0 < len(self.targets) <= 2, "LineAnnotation must have one or two targets"
        return np.array(
            [
                self.targets[0].calculate_display_coordinates_at_resiude(
                    self.indices[0]
                ),
                self.targets[1].calculate_display_coordinates_at_resiude(
                    self.indices[1]
                )
                if len(self.targets) == 2
                else self.targets[0].calculate_display_coordinates_at_resiude(
                    self.indices[1]
                ),
            ]