# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from flatprot.style import StyleManager, ElementStyle, StyleType


class SceneElement(ABC):
    """A class to represent a scene element of a protein."""

    def __init__(
        self,
        start: int,
        end: int,
        chain_id: str,
        metadata: dict,
        style_manager: StyleManager,
        style_type: StyleType,
    ):
        self._start = start
        self._end = end
        self._chain_id = chain_id
        self._metadata = metadata
        self._style_manager = style_manager
        self._style_type = style_type

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def style(self) -> ElementStyle:
        return self._style_manager.get_style(self._style_type)

    @property
    def style_type(self) -> StyleType:
        return self._style_type

    @abstractmethod
    def display_coordinates(self) -> np.ndarray:
        """Calculate the coordinates of the scene element."""
        pass


class SceneGroup(SceneElement):
    def __init__(
        self,
        id: str,
        metadata: dict = {},
        transforms: dict = {},
        style_manager: Optional[StyleManager] = None,
        style_type: Optional[StyleType] = None,
    ):
        super().__init__(metadata, style_manager, style_type)
        self.id = id
        self._elements: list[SceneElement] = []
        self._transforms = transforms
        # Track parent relationships
        self._parent: Optional[SceneGroup] = None

    @property
    def transforms(self) -> dict:
        return self._transforms

    def add_element(self, element: SceneElement) -> None:
        if isinstance(element, SceneElement):
            if hasattr(element, "_parent"):
                element._parent = self
            self._elements.append(element)
        else:
            raise TypeError("Element must be an instance of SceneElement")

    def remove_element(self, element: SceneElement) -> None:
        """Remove an element from this group.

        Args:
            element: The element to remove
        """
        if element in self._elements:
            self._elements.remove(element)
            if hasattr(element, "_parent"):
                element._parent = None

    def __iter__(self):
        return iter(self._elements)

    def display_coordinates(self) -> Optional[np.ndarray]:
        """Calculate combined coordinates of all child elements.

        Returns:
            Combined coordinates array or None if group has no displayable elements
        """
        # Collect coordinates from all children that have them
        coords = [
            elem.display_coordinates()
            for elem in self._elements
            if elem.display_coordinates() is not None
        ]

        if not coords:
            return None

        return np.concatenate(coords, axis=0)
