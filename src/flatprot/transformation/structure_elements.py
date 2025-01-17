# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from flatprot.core.secondary import SecondaryStructure, SecondaryStructureType
from .base import Transformer, TransformParameters
from .utils import (
    TransformationMatrix,
    apply_transformation,
    calculate_inertia_transformation,
)


@dataclass
class StructureElementsTransformerParameters:
    """Parameters for structure elements projection calculation."""

    non_structure_weight: float = 0.5
    structure_weight: float = 1.0


@dataclass
class StructureElementsTransformParameters(TransformParameters):
    """Parameters for structure elements transformation calculation."""

    structure_elements: list[SecondaryStructure]


class StructureElementsTransformer(Transformer):
    """Projects using structure elements."""

    def __init__(
        self, parameters: Optional[StructureElementsTransformerParameters] = None
    ):
        super().__init__()
        self.parameters = parameters or StructureElementsTransformerParameters()

    def _calculate_transformation(
        self,
        coordinates: np.ndarray,
        parameters: Optional[StructureElementsTransformParameters] = None,
    ) -> TransformationMatrix:
        """Calculate transformation matrix for given coordinates."""
        if parameters is None or not parameters.structure_elements:
            # Use default weight for all coordinates if no structure elements provided
            weights = np.ones(len(coordinates)) * self.parameters.non_structure_weight
        else:
            # Initialize with non-structure weight
            weights = np.ones(len(coordinates)) * self.parameters.non_structure_weight
            # Set higher weight for structure elements
            for element in parameters.structure_elements:
                if element.type != SecondaryStructureType.COIL:
                    weights[element.start : element.end] = (
                        self.parameters.structure_weight
                    )

        return calculate_inertia_transformation(coordinates, weights)

    def _apply_cached_transformation(
        self,
        coordinates: np.ndarray,
        cached_transformation: TransformationMatrix,
        parameters: Optional[StructureElementsTransformParameters] = None,
    ) -> np.ndarray:
        """Apply cached transformation to coordinates."""
        return apply_transformation(coordinates, cached_transformation)
