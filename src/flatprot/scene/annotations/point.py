# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .base import Annotation
import numpy as np

from flatprot.style import StyleType


class PointAnnotation(Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style_type = StyleType.POINT_ANNOTATION

    def display_coordinates(self) -> np.ndarray:
        assert len(self.indices) == 1, "PointAnnotation must have exactly one index"
        assert len(self.targets) == 1, "PointAnnotation must have exactly one target"
        return [
            self.targets[0].calculate_display_coordinates_at_resiude(self.indices[0])
        ]
