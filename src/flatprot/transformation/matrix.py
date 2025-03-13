# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .base import BaseTransformer, TransformParameters
from .utils import TransformationMatrix, apply_transformation


@dataclass
class MatrixTransformParameters(TransformParameters):
    """Parameters for matrix-based transformation."""

    matrix: TransformationMatrix


class MatrixTransformer(BaseTransformer):
    """Projects coordinates using a provided rotation matrix and translation vector.

    This projector applies a fixed projection matrix provided at initialization.
    """

    def __init__(self):
        """Initialize with a fixed transformation matrix.

        Args:
            projection_matrix: The ProjectionMatrix to use for all projections
        """
        super().__init__()

    def _calculate_transformation(
        self,
        coordinates: np.ndarray,
        parameters: Optional[MatrixTransformParameters] = None,
    ) -> TransformationMatrix:
        """Return the fixed projection matrix.

        Since this projector uses a fixed projection matrix provided at initialization,
        this method simply returns the cached transformation.
        """
        self._cached_transformation = parameters.matrix
        return self._cached_transformation

    def _apply_cached_transformation(
        self,
        coordinates: np.ndarray,
        cached_transformation: TransformationMatrix,
        parameters: Optional[MatrixTransformParameters] = None,
    ) -> np.ndarray:
        """Apply the fixed transformation to coordinates."""
        return apply_transformation(coordinates, cached_transformation)
