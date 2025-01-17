# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from flatprot.core import Structure
from flatprot.visualization.element import VisualizationElement


class CoordinateManager:
    """Manages different types of coordinates (original, transformed, projected)."""

    def __init__(self):
        self.coordinates: dict[tuple[int, int], np.ndarray] = {}
        self.transformed: dict[tuple[int, int], np.ndarray] = {}
        self.projected: dict[tuple[int, int], np.ndarray] = {}

    def add(
        self, start: int, end: int, coords: np.ndarray, coord_type: str = "coordinates"
    ) -> None:
        """Add coordinates for a range."""
        if end < start:
            raise ValueError("End must be >= start")
        target_map = getattr(self, coord_type)
        target_map[start, end] = np.asarray(coords)

    def get(
        self, start: int, end: int | None = None, coord_type: str = "coordinates"
    ) -> np.ndarray:
        """Get coordinates for an index or range."""
        target_map = getattr(self, coord_type)

        if end is None:
            for (s, e), coords in target_map.items():
                if s <= start <= e:
                    return coords[start - s]
            raise KeyError(f"No coordinates at index {start}")

        result = []
        for (s, e), coords in target_map.items():
            o_start = max(start, s)
            o_end = min(end, e)
            if o_start <= o_end:
                result.append(coords[o_start - s : o_end - s + 1])

        if not result:
            raise KeyError(f"No coordinates in range [{start}, {end}]")
        return np.concatenate(result)

    @staticmethod
    def get_bounds(coords: np.ndarray) -> np.ndarray:
        """Get bounding box [min_x, min_y, max_x, max_y]."""
        return np.array(
            [
                coords[:, 0].min(),
                coords[:, 1].min(),
                coords[:, 0].max(),
                coords[:, 1].max(),
            ]
        )


class VisualizationManager:
    """Manages visualization elements and their properties."""

    def __init__(self):
        self.elements: dict[tuple[int, int], VisualizationElement] = {}
        self.coord_manager = CoordinateManager()

    def add_element(
        self, start: int, end: int, element: VisualizationElement, coords: np.ndarray
    ) -> None:
        """Add a visualization element with its coordinates."""
        self.elements[start, end] = element
        self.coord_manager.add(start, end, coords)

    def get_element(self, index: int) -> VisualizationElement:
        """Get visualization element at index."""
        for (start, end), element in self.elements.items():
            if start <= index <= end:
                return element
        raise KeyError(f"No visualization element at index {index}")
