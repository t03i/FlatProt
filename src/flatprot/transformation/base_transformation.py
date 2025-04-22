# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Optional, TypeVar, Generic
from dataclasses import dataclass

from .transformation_matrix import TransformationMatrix


@dataclass
class BaseTransformationParameters:
    """Base class for parameters configuring a transformation instance."""

    pass


P = TypeVar("P", bound=BaseTransformationParameters)


# --- Runtime Arguments (for transform method) ---
@dataclass
class BaseTransformationArguments:
    """Base class for arguments specific to a single transform() call."""

    pass


A = TypeVar("A", bound=BaseTransformationArguments)


class BaseTransformation(ABC, Generic[P, A]):
    """Base class for all transformations using template method pattern."""

    def __init__(self, parameters: Optional[P]):
        self._cached_transformation: Optional[TransformationMatrix] = None
        self.parameters = parameters

    def transform(
        self,
        coordinates: np.ndarray,
        arguments: Optional[A],
    ) -> np.ndarray:
        """Transform 3D coordinates.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates

        Returns:
            Array of shape (N, 3) containing transformed coordinates
        """
        if self._cached_transformation is None:
            self._cached_transformation = self._calculate_transformation_matrix(
                coordinates, arguments
            )

        transformed = self._apply_transformation(coordinates)
        return transformed

    @abstractmethod
    def _calculate_transformation_matrix(
        self,
        coordinates: np.ndarray,
        arguments: Optional[A],
    ) -> TransformationMatrix:
        """Calculate transformation matrix for given coordinates.

        Args:
            coordinates: 3D coordinates to project

        Returns:
            Transformation matrix
        """
        pass

    def _apply_transformation(
        self,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Applies the standard transformation: (R @ X) + T."""
        return self._cached_transformation.apply(coordinates)

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
