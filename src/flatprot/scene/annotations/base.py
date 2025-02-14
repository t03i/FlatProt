# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import numpy as np

from ..scene import SceneElement
from flatprot.style import StyleManager, StyleType
from ..structure import StructureSceneElement


class AnnotationElement(SceneElement):
    """Base class for annotations"""

    def __init__(
        self,
        positions: list[int],
        elements: list[StructureSceneElement],
        metadata: dict,
        style_manager: StyleManager,
        style_type: StyleType,
    ):
        super().__init__(metadata, style_manager, style_type)
        self._positions = positions
        self._elements = elements

    @abstractmethod
    def display_coordinates(self) -> np.ndarray:
        """Calculate the coordinates of the annotation element"""
        pass
