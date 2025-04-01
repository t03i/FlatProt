# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
import numpy as np
from typing import Optional, Dict

from .residue import ResidueCoordinate, ResidueRange, ResidueRangeSet
from .error import CoordinateError


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
        self._coordinates: Dict[
            CoordinateType, Dict[ResidueCoordinate, np.ndarray]
        ] = {}

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
        target_map = self._coordinates[coord_type]
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
        target_map = self._coordinates[coord_type]

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
        return coord_type in self._coordinates and bool(self._coordinates[coord_type])

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
        segments = sorted(self._coordinates[coord_type].items(), key=lambda x: x[0][0])

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

    def add_range(
        self,
        residue_range: ResidueRange,
        coords: np.ndarray,
        coord_type: CoordinateType,
    ) -> None:
        """Add coordinates for a continuous range of residues."""
        if coords.shape[0] != len(residue_range):
            raise CoordinateError(
                f"Coordinate array length {coords.shape[0]} does not match range length {len(residue_range)}"
            )

        for i, residue in enumerate(residue_range):
            self.add(residue, coords[i], coord_type)

    def add_range_set(
        self,
        range_set: ResidueRangeSet,
        coords: np.ndarray,
        coord_type: CoordinateType,
    ) -> None:
        """Add coordinates for multiple ranges."""
        if coords.shape[0] != len(range_set):
            raise CoordinateError(
                f"Coordinate array length {coords.shape[0]} does not match total range length {len(range_set)}"
            )

        offset = 0
        for range_ in range_set.ranges:
            length = len(range_)
            self.add_range(range_, coords[offset : offset + length], coord_type)
            offset += length

    def get_range(
        self,
        residue_range: ResidueRange,
        coord_type: CoordinateType,
    ) -> np.ndarray:
        """Get coordinates for a continuous range."""
        coords = []
        for residue in residue_range:
            coord = self.get(residue, coord_type)
            if coord is None:
                raise CoordinateError(f"No coordinates found for {residue}")
            coords.append(coord)
        return np.array(coords)

    def get_range_set(
        self,
        range_set: ResidueRangeSet,
        coord_type: CoordinateType,
    ) -> np.ndarray:
        """Get coordinates for multiple ranges."""
        coords = []
        for range_ in range_set.ranges:
            coords.append(self.get_range(range_, coord_type))
        return np.concatenate(coords)
