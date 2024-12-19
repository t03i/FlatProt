# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .projector import Projector, ProjectionScope
from ..structure.components import Structure
from ..structure.secondary import SecondaryStructureType
from .utils import ProjectionMatrix

import flatprot.projection.utils as utils


@dataclass
class StructureElementsParameters:
    """Parameters for structure elements projection calculation."""

    non_structure_weight: float = 0.5
    structure_weight: float = 1.0


class InertiaProjector(Projector):
    """Projects using inertia-based calculation with residue weights."""

    def __init__(
        self,
        parameters: Optional[StructureElementsParameters] = None,
        scope: ProjectionScope = ProjectionScope.STRUCTURE,
    ):
        super().__init__(scope=scope)
        self.parameters = parameters or StructureElementsParameters()

    def _calculate_projection(
        self,
        structure: Structure,
        coordinates: np.ndarray,
        chain_id: Optional[str] = None,
    ) -> ProjectionMatrix:
        """Calculate projection matrix for given coordinates."""
        # Get residues based on scope
        if chain_id is None:
            residues = []
            for chain in structure.values():
                residues.extend(chain.residues)
        else:
            residues = structure[chain_id].residues

        # Calculate weights
        weights = np.ones(len(coordinates)) * self.parameters.non_structure_weight

        # Set weights for residues in secondary structure elements
        if chain_id is None:
            for chain in structure.values():
                for ss in chain.secondary_structure:
                    if ss.type in [
                        SecondaryStructureType.HELIX,
                        SecondaryStructureType.SHEET,
                    ]:
                        weights[ss.start : ss.end + 1] = (
                            self.parameters.structure_weight
                        )
        else:
            chain = structure[chain_id]
            for ss in chain.secondary_structure:
                if ss.type in [
                    SecondaryStructureType.HELIX,
                    SecondaryStructureType.SHEET,
                ]:
                    weights[ss.start : ss.end + 1] = self.parameters.structure_weight

        return utils.calculate_inertia_projection(coordinates, weights)

    def _apply_cached_projection(
        self, chain, cached_projection: ProjectionMatrix
    ) -> np.ndarray:
        """Apply cached projection to chain coordinates."""
        coordinates = chain.coordinates
        centered = coordinates - cached_projection.translation
        rotated = np.dot(centered, cached_projection.rotation)
        return rotated[:, :2]  # Take only x,y coordinates
