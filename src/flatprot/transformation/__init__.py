# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from .base import BaseTransformation, TransformationParameters
from .inertia import (
    InertiaTransformer,
    InertiaTransformationParameters,
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
    "BaseTransformation",
    "TransformationParameters",
    "InertiaTransformer",
    "InertiaTransformationParameters",
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
