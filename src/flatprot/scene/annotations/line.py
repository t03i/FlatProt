# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .base import Annotation
import numpy as np

from flatprot.style import StyleType


class LineAnnotation(Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style_type = StyleType.LINE_ANNOTATION

    def display_coordinates(self) -> np.ndarray:
        assert len(self.indices) == 2, "LineAnnotation must have exactly two indices"
        assert len(self.targets) == 2, "LineAnnotation must have exactly two targets"
        return np.array(
            [
                self.targets[0].calculate_display_coordinates_at_resiude(
                    self.indices[0]
                ),
                self.targets[1].calculate_display_coordinates_at_resiude(
                    self.indices[1]
                ),
            ]
        )
