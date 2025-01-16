# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectionParameters:
    """Base parameters for projections."""

    view_direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    up_vector: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    scale: float = 1.0
    center: bool = True


class Projector(ABC):
    """Base class for all projections."""

    @abstractmethod
    def project(
        self, coordinates: np.ndarray, parameters: Optional[ProjectionParameters] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D coordinates to 2D space.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates
            parameters: Optional projection parameters

        Returns:
            Array of shape (N, 2) containing projected coordinates
            Array of shape (N,) containing depth values
        """
        pass
