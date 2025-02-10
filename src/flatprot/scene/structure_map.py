# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .structure import VisualizationElement


class StructureMap:
    """A class to represent a structure map of a protein."""

    def __init__(self):
        self.__residue_map: dict[tuple[int, int], VisualizationElement] = {}
        self.__sorted_ranges: list[
            tuple[int, int]
        ] = []  # Keep ranges sorted for efficient lookup

    def add_element(self, start: int, end: int, element: VisualizationElement):
        """Add a visualization element for a range of residues."""
        if start > end:
            raise ValueError(f"Invalid range: start ({start}) > end ({end})")

        # Check for overlapping ranges
        for existing_start, existing_end in self.__residue_map.keys():
            if not (end < existing_start or start > existing_end):
                raise ValueError(
                    f"Overlapping range: [{start}, {end}] with [{existing_start}, {existing_end}]"
                )

        self.__residue_map[(start, end)] = element
        self.__sorted_ranges = sorted(self.__residue_map.keys(), key=lambda x: x[0])

    def get_element(self, residue: int) -> VisualizationElement:
        """Get the visualization element for a specific residue using binary search."""
        left, right = 0, len(self.__sorted_ranges) - 1

        while left <= right:
            mid = (left + right) // 2
            start, end = self.__sorted_ranges[mid]

            if start <= residue <= end:
                return self.__residue_map[(start, end)]
            elif residue < start:
                right = mid - 1
            else:
                left = mid + 1

        raise ValueError(f"No element found for residue {residue}")

    def get_ranges(self) -> list[tuple[tuple[int, int], VisualizationElement]]:
        """Get all ranges and their elements in sorted order."""
        return [(r, self.__residue_map[r]) for r in self.__sorted_ranges]

    def get_coordinate_for_residue(self, residue: int) -> np.ndarray:
        element = self.get_element(residue)
        return element.transform_coordinates(residue)
