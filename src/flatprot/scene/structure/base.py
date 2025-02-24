# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from typing import Optional

import numpy as np

from ..elements import SceneElement, SceneGroup
from flatprot.style import StyleManager, StyleType


class StructureSceneElement(SceneElement):
    """A scene element that represents a structure"""

    def __init__(
        self,
        canvas_coordinates: np.ndarray,
        style_manager: StyleManager,
        style_type: StyleType,
        metadata: Optional[dict] = None,
    ):
        super().__init__(style_manager, style_type, metadata)
        self._coordinates = canvas_coordinates
        self._display_coordinates = self.calculate_display_coordinates()

    @property
    def display_coordinates(self) -> np.ndarray:
        return self._display_coordinates

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @abstractmethod
    def calculate_display_coordinates(self) -> np.ndarray:
        """Calculate the coordinates of the structure component"""
        pass

    @abstractmethod
    def display_coordinates_at_position(self, position: int) -> np.ndarray:
        """Get the display coordinates of the structure component"""
        pass
