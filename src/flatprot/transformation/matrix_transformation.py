# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np

from .base_transformation import (
    BaseTransformation,
    BaseTransformationParameters,
)
from .transformation_matrix import TransformationMatrix


@dataclass
class MatrixTransformParameters(BaseTransformationParameters):
    """Parameters for matrix-based transformation."""

    matrix: TransformationMatrix


class MatrixTransformer(BaseTransformation[MatrixTransformParameters, None]):
    """Projects coordinates using a provided rotation matrix and translation vector.

    This projector applies a fixed projection matrix provided at initialization.
    """

    def __init__(self, parameters: MatrixTransformParameters):
        """Initialize with a fixed transformation matrix.

        Args:
            projection_matrix: The ProjectionMatrix to use for all projections
        """
        super().__init__(parameters)
        self._cached_transformation = parameters.matrix

    def _calculate_transformation_matrix(
        self,
        coordinates: np.ndarray,
    ) -> TransformationMatrix:
        """Return the fixed projection matrix.

        Since this projector uses a fixed projection matrix provided at initialization,
        this method simply returns the cached transformation.
        """
        return self._cached_transformation
