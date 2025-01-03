# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from flatprot.structure.secondary import SecondaryStructure, SecondaryStructureType
from .projector import Projector, ProjectionParameters
from .utils import ProjectionMatrix

import flatprot.projection.utils as utils


@dataclass
class StructureElementsParameters:
    """Parameters for structure elements projection calculation."""

    non_structure_weight: float = 0.5
    structure_weight: float = 1.0


@dataclass
class StructureElementsProjectionParameters(ProjectionParameters):
    """Parameters for structure elements projection calculation."""

    structure_elements: list[SecondaryStructure]


class StructureElementsProjector(Projector):
    """Projects using structure elements."""

    def __init__(self, parameters: Optional[StructureElementsParameters] = None):
        super().__init__()
        self.parameters = parameters or StructureElementsParameters()

    def _calculate_projection(
        self,
        coordinates: np.ndarray,
        parameters: Optional[StructureElementsProjectionParameters] = None,
    ) -> ProjectionMatrix:
        """Calculate projection matrix for given coordinates."""
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

        return utils.calculate_inertia_projection(coordinates, weights)

    def _apply_cached_projection(
        self,
        coordinates: np.ndarray,
        cached_projection: ProjectionMatrix,
        parameters: Optional[StructureElementsProjectionParameters] = None,
    ) -> np.ndarray:
        """Apply cached projection to coordinates."""
        return utils.apply_projection(coordinates, cached_projection)
