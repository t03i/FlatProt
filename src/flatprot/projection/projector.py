# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .utils import TransformationMatrix


@dataclass
class ProjectionParameters:
    """Parameters for the projector."""

    pass


class Projector(ABC):
    """Base class for all projectors using template method pattern."""

    def __init__(self):
        self._cached_projection: Optional[TransformationMatrix] = None

    def project(
        self, coordinates: np.ndarray, parameters: Optional[ProjectionParameters] = None
    ) -> np.ndarray:
        """Project 3D coordinates to 2D space while preserving Z-coordinate.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates

        Returns:
            Array of shape (N, 3) containing projected coordinates (x, y, original_z)
        """
        if self._cached_projection is None:
            self._cached_projection = self._calculate_projection(
                coordinates, parameters
            )

        projected = self._apply_cached_projection(
            coordinates, self._cached_projection, parameters
        )
        # Preserve the original z-coordinate for depth sorting
        return projected

    @abstractmethod
    def _calculate_projection(
        self, coordinates: np.ndarray, parameters: Optional[ProjectionParameters] = None
    ) -> TransformationMatrix:
        """Calculate projection matrix for given coordinates.

        Args:
            coordinates: 3D coordinates to project

        Returns:
            Projection matrix
        """
        pass

    @abstractmethod
    def _apply_cached_projection(
        self,
        coordinates: np.ndarray,
        cached_projection: TransformationMatrix,
        parameters: Optional[ProjectionParameters] = None,
    ) -> np.ndarray:
        """Apply cached projection to coordinates, returning 2D coordinates."""
        pass

    def save(self, path: Path) -> None:
        """Saves projection parameters."""
        if path.suffix != ".npz":
            raise ValueError(
                f"Projection must be saved as .npz file, got '{path.suffix}'"
            )
        np.savez(path, projection=self._cached_projection.to_array())

    def load(self, path: Path) -> None:
        """Loads projection parameters."""
        if path.suffix != ".npz":
            raise ValueError(
                f"Projection must be loaded from .npz file, got '{path.suffix}'"
            )
        loaded = np.load(path)
        self._cached_projection = TransformationMatrix.from_array(loaded["projection"])
