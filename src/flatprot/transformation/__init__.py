# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .base import BaseTransformer, TransformParameters
from .inertia import (
    InertiaTransformer,
    InertiaTransformerParameters,
    InertiaTransformParameters,
)
from .structure_elements import (
    StructureElementsTransformer,
    StructureElementsTransformerParameters,
    StructureElementsTransformParameters,
)
from .matrix import MatrixTransformer, MatrixTransformParameters
from .utils import TransformationMatrix
from flatprot.core.error import FlatProtError

__all__ = [
    "BaseTransformer",
    "TransformParameters",
    "InertiaTransformer",
    "InertiaTransformerParameters",
    "InertiaTransformParameters",
    "StructureElementsTransformer",
    "StructureElementsTransformerParameters",
    "StructureElementsTransformParameters",
    "TransformationMatrix",
    "MatrixTransformer",
    "MatrixTransformParameters",
    "TransformationError",
]


# Transformation-related errors
class TransformationError(FlatProtError):
    """Exception raised when a transformation operation fails."""

    def __init__(self, message: str):
        super().__init__(f"Transformation error: {message}")
