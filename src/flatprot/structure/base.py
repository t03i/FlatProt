# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np

from .residue import Residue


class StructureComponent:
    """
    Base class for a structure component.
    """

    def __init__(
        self,
        residues: list[Residue],
        index: np.ndarray,
        coordinates: np.ndarray,
    ):
        self.__residues = residues
        self.__index = index
        self.__coordinates = coordinates
        self.__sub_components: list[StructureComponent] = []

    def add_sub_component(self, sub_component: "StructureComponent") -> None:
        self.__sub_components.append(sub_component)

    @property
    def coordinates(self) -> np.ndarray:
        return self.__coordinates

    @property
    def residues(self) -> list[Residue]:
        return self.__residues

    @property
    def index(self) -> np.ndarray:
        return self.__index

    def get_coordinates_for_index(self, index: int) -> np.array:
        pos = np.where(self.index == index)[0]
        return self.coordinates[pos, :]

    def transform(
        self, rotation: np.ndarray, translation: np.ndarray
    ) -> "StructureComponent":
        transformed_coords = rotation @ self.coordinates + translation
        comp = StructureComponent(
            residues=self.residues, index=self.index, coordinates=transformed_coords
        )
        for sub_comp in self._sub_components:
            transformed_sub = type(sub_comp)(
                parent=comp, start_idx=sub_comp.start_idx, end_idx=sub_comp.end_idx
            )
            comp.add_sub_component(transformed_sub)
        return comp


class SubStructureComponent(StructureComponent):
    def __init__(self, parent: StructureComponent, start_idx: int, end_idx: int):
        self.__sub_components = []
        self.__parent = parent
        self.__start_idx = start_idx
        self.__end_idx = end_idx

    @property
    def coordinates(self) -> np.ndarray:
        return self.__parent.coordinates[self.__start_idx : self.__end_idx + 1, :]

    @property
    def residues(self) -> list[Residue]:
        return self.__parent.residues[self.__start_idx : self.__end_idx + 1]

    @property
    def index(self) -> np.ndarray:
        return self.__parent.index[self.__start_idx : self.__end_idx + 1]

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement this method")
