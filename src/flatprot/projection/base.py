# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar, Generic

from pydantic import BaseModel, Field
from numpydantic import NDArray, Shape


class BaseProjectionParameters(BaseModel):
    """Base parameters for projections."""

    view_direction: NDArray[Shape["3"], float] = Field(
        default_factory=lambda: np.array([0, 0, 1])
    )
    up_vector: NDArray[Shape["3"], float] = Field(
        default_factory=lambda: np.array([0, 1, 0])
    )
    center: bool = True


P = TypeVar("P", bound=BaseProjectionParameters)


class BaseProjection(ABC, Generic[P]):
    """Base class for all projections."""

    @abstractmethod
    def project(self, coordinates: np.ndarray, parameters: P) -> np.ndarray:
        """Project 3D coordinates to 2D space.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates
            parameters: Optional projection parameters

        Returns:
            A NumPy array of shape (N, 3), where N is the number of coordinates.
            Column 0: X canvas coordinate (float)
            Column 1: Y canvas coordinate (float)
            Column 2: Depth value (float, representing scaled Z for visibility/layering)
        """
        pass
