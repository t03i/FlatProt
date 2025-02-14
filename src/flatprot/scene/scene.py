# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np

from flatprot.style import StyleManager, ElementStyle, StyleType


class SceneElement(ABC):
    """A class to represent a scene element of a protein."""

    def __init__(
        self,
        metadata: dict,
        style_manager: StyleManager,
        style_type: StyleType,
    ):
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
    ):
        super().__init__(metadata, None, None)
        self.id = id
        self._elements: list[SceneElement] = []
        self._transforms = transforms

    @property
    def transforms(self) -> dict:
        return self._transforms

    def add_element(self, element: SceneElement):
        self._elements.append(element)

    def __iter__(self):
        return iter(self._elements)

    def display_coordinates(self) -> np.ndarray:
        return None


class Scene:
    """A class to represent a scene of a protein."""

    def __init__(self):
        self.__elements: list[SceneElement] = []

    def add_element(self, element: SceneElement):
        self.__elements.append(element)

    def __iter__(self):
        return iter(self.__elements)

    def __getitem__(self, index: int) -> SceneElement:
        return self.__elements[index]
