# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .utils import TransformationMatrix


@dataclass
class TransformParameters:
    """Parameters for the transformation."""

    pass


class BaseTransformer(ABC):
    """Base class for all transformations using template method pattern."""

    def __init__(self):
        self._cached_transformation: Optional[TransformationMatrix] = None

    def transform(
        self,
        coordinates: np.ndarray,
        parameters: Optional[TransformParameters] = None,
    ) -> np.ndarray:
        """Transform 3D coordinates.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates

        Returns:
            Array of shape (N, 3) containing transformed coordinates
        """
        if self._cached_transformation is None:
            self._cached_transformation = self._calculate_transformation(
                coordinates, parameters
            )

        transformed = self._apply_cached_transformation(
            coordinates, self._cached_transformation, parameters
        )
        return transformed

    @abstractmethod
    def _calculate_transformation(
        self,
        coordinates: np.ndarray,
        parameters: Optional[TransformParameters] = None,
    ) -> TransformationMatrix:
        """Calculate transformation matrix for given coordinates.

        Args:
            coordinates: 3D coordinates to project

        Returns:
            Transformation matrix
        """
        pass

    @abstractmethod
    def _apply_cached_transformation(
        self,
        coordinates: np.ndarray,
        cached_transformation: TransformationMatrix,
        parameters: Optional[TransformParameters] = None,
    ) -> np.ndarray:
        """Apply cached transformation to coordinates."""
        pass

    def save(self, path: Path) -> None:
        """Saves transformation parameters."""
        if path.suffix != ".npz":
            raise ValueError(
                f"Transformation must be saved as .npz file, got '{path.suffix}'"
            )
        np.savez(path, transformation=self._cached_transformation.to_array())

    def load(self, path: Path) -> None:
        """Loads transformation parameters."""
        if path.suffix != ".npz":
            raise ValueError(
                f"Transformation must be loaded from .npz file, got '{path.suffix}'"
            )
        loaded = np.load(path)
        self._cached_transformation = TransformationMatrix.from_array(
            loaded["transformation"]
        )
