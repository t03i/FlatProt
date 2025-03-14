# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from collections import defaultdict
import numpy as np
from typing import Optional


class CoordinateType(Enum):
    COORDINATES = "coordinates"
    TRANSFORMED = "transformed"
    CANVAS = "canvas"
    DEPTH = "depth"


class CoordinateManager:
    """Manages different types of coordinates (original, transformed, projected).

    This class provides methods to store, retrieve, and check for the existence of
    different types of coordinate data in a protein structure visualization.
    """

    def __init__(self) -> None:
        """Initialize an empty coordinate manager."""
        self.coordinates: dict[
            CoordinateType, dict[tuple[int, int], np.ndarray]
        ] = defaultdict(dict)

    def add(
        self,
        start: int,
        end: int,
        coords: np.ndarray,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> None:
        """Add coordinates for a range.

        Args:
            start: Start index of the range (inclusive)
            end: End index of the range (inclusive)
            coords: Coordinate array to add
            coord_type: Type of coordinates to add

        Raises:
            ValueError: If end is less than start
        """
        if end < start:
            raise ValueError("End must be >= start")
        target_map = self.coordinates[coord_type]
        target_map[start, end] = np.asarray(coords)

    def get(
        self,
        start: int,
        end: Optional[int] = None,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> np.ndarray:
        """Get coordinates for an index or range.

        Args:
            start: Start index of the range (inclusive)
            end: End index of the range (inclusive); if None, return only the
                 coordinate at index 'start'
            coord_type: Type of coordinates to retrieve

        Returns:
            Numpy array containing the requested coordinates

        Raises:
            KeyError: If no coordinates exist for the specified range
        """
        target_map = self.coordinates[coord_type]

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

    def has_type(self, coord_type: CoordinateType) -> bool:
        """Check if coordinates of a specific type exist.

        Args:
            coord_type: Type of coordinates to check for

        Returns:
            True if coordinates of the specified type exist, False otherwise
        """
        return coord_type in self.coordinates and bool(self.coordinates[coord_type])

    def get_all(self, coord_type: CoordinateType) -> np.ndarray:
        """Get all coordinates of a specific type.

        This function concatenates all coordinate segments of the specified type.

        Args:
            coord_type: Type of coordinates to retrieve

        Returns:
            Numpy array containing all coordinates of the specified type

        Raises:
            KeyError: If no coordinates of the specified type exist
        """
        if not self.has_type(coord_type):
            raise KeyError(f"No coordinates of type {coord_type.value} exist")

        # Collect all coordinate segments, sorted by start index
        segments = sorted(self.coordinates[coord_type].items(), key=lambda x: x[0][0])

        # Concatenate all segments
        if not segments:
            return np.array([])

        return np.concatenate([coords for _, coords in segments])

    @staticmethod
    def get_bounds(coords: np.ndarray) -> np.ndarray:
        """Get bounding box [min_x, min_y, max_x, max_y].

        Args:
            coords: Coordinate array of shape (N, 2) or (N, 3)

        Returns:
            Numpy array with bounding box coordinates [min_x, min_y, max_x, max_y]
        """
        return np.array(
            [
                coords[:, 0].min(),
                coords[:, 1].min(),
                coords[:, 0].max(),
                coords[:, 1].max(),
            ]
        )
