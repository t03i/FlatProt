# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .projector import Projector, ProjectionParameters
from .utils import TransformationMatrix, apply_projection


@dataclass
class MatrixProjectionParameters(ProjectionParameters):
    """Parameters for matrix-based projection."""

    pass


class MatrixProjector(Projector):
    """Projects coordinates using a provided rotation matrix and translation vector."""

    def __init__(self, rotation_matrix: TransformationMatrix):
        self._cached_projection = rotation_matrix

    def _calculate_projection(
        self,
        coordinates: np.ndarray,
        parameters: Optional[MatrixProjectionParameters] = None,
    ) -> TransformationMatrix:
        """Calculate projection matrix from provided parameters."""

        return self._cached_projection

    def _apply_cached_projection(
        self,
        coordinates: np.ndarray,
        cached_projection: TransformationMatrix,
        parameters: Optional[MatrixProjectionParameters] = None,
    ) -> np.ndarray:
        """Apply cached projection to coordinates."""
        return apply_projection(coordinates, cached_projection)
