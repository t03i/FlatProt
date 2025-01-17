# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import numpy as np

from .structure.structure import Structure


class CoordinateType(Enum):
    COORDINATES = "coordinates"
    TRANSFORMED = "transformed"
    CANVAS = "canvas"
    DEPTH = "depth"


class CoordinateManager:
    """Manages different types of coordinates (original, transformed, projected)."""

    def __init__(self):
        self.coordinates: dict[CoordinateType, dict[tuple[int, int], np.ndarray]] = {
            CoordinateType.COORDINATES: {},
            CoordinateType.TRANSFORMED: {},
            CoordinateType.PROJECTED: {},
        }

    def add(
        self,
        start: int,
        end: int,
        coords: np.ndarray,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> None:
        """Add coordinates for a range."""
        if end < start:
            raise ValueError("End must be >= start")
        target_map = self.coordinates[coord_type]
        target_map[start, end] = np.asarray(coords)

    def get(
        self,
        start: int,
        end: int | None = None,
        coord_type: CoordinateType = CoordinateType.COORDINATES,
    ) -> np.ndarray:
        """Get coordinates for an index or range."""
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
