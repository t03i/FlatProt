# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .projector import Projector
from .utils import ProjectionMatrix, ProjectionParameters

import flatprot.projection.utils as utils


@dataclass
class StructureElementsParameters:
    """Parameters for structure elements projection calculation."""

    non_structure_weight: float = 0.5
    structure_weight: float = 1.0


class InertiaProjector(Projector):
    """Projects using inertia-based calculation with residue weights."""

    def __init__(self, parameters: Optional[StructureElementsParameters] = None):
        super().__init__()
        self.parameters = parameters or StructureElementsParameters()

    def _calculate_projection(
        self,
        coordinates: np.ndarray,
        parameters: Optional[ProjectionParameters] = None,
    ) -> ProjectionMatrix:
        """Calculate projection matrix for given coordinates."""
        # Use default weight for all coordinates
        weights = np.ones(len(coordinates)) * self.parameters.non_structure_weight
        return utils.calculate_inertia_projection(coordinates, weights)

    def _apply_cached_projection(
        self,
        coordinates: np.ndarray,
        cached_projection: ProjectionMatrix,
        parameters: Optional[ProjectionParameters] = None,
    ) -> np.ndarray:
        """Apply cached projection to coordinates."""
        return utils.apply_projection(coordinates, cached_projection)
