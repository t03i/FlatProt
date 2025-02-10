# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from . import VisualizationElement


class StructureMap:
    """A class to represent a structure map of a protein."""

    def __init__(self):
        self.__residue_map: dict[tuple[int, int], VisualizationElement] = {}

    def add_element(self, start: int, end: int, element: VisualizationElement):
        self.__residue_map[(start, end)] = element

    def get_element(self, residue: int) -> VisualizationElement:
        for (start, end), element in self.__residue_map.items():
            if start <= residue <= end:
                return element
        raise ValueError(f"No element found for residue {residue}")

    def get_coordinate_for_residue(self, residue: int) -> np.ndarray:
        element = self.get_element(residue)
        return element.transform_coordinates(residue)
