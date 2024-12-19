# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import dict, Optional

from .projector import Projector, ProjectionScope
from ..structure.components import Structure
from .utils import ProjectionMatrix
import flatprot.projection.utils as utils


@dataclass
class InertiaParameters:
    """Parameters for inertia-based projection calculation."""

    residue_weights: dict[str, float]  # Maps residue type to weight
    use_weights: bool = True

    @classmethod
    def default(cls) -> "InertiaParameters":
        """Creates default parameters using standard amino acid weights."""
        return cls(
            residue_weights={
                "ALA": 89.1,
                "ARG": 174.2,
                "ASN": 132.1,
                "ASP": 133.1,
                "CYS": 121.2,
                "GLN": 146.2,
                "GLU": 147.1,
                "GLY": 75.1,
                "HIS": 155.2,
                "ILE": 131.2,
                "LEU": 131.2,
                "LYS": 146.2,
                "MET": 149.2,
                "PHE": 165.2,
                "PRO": 115.1,
                "SER": 105.1,
                "THR": 119.1,
                "TRP": 204.2,
                "TYR": 181.2,
                "VAL": 117.1,
            }
        )


class InertiaProjector(Projector):
    """Projects using inertia-based calculation with residue weights."""

    def __init__(
        self,
        parameters: Optional[InertiaParameters] = None,
        scope: ProjectionScope = ProjectionScope.STRUCTURE,
    ):
        super().__init__(scope=scope)
        self.parameters = parameters or InertiaParameters.default()

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
        if not self.parameters.use_weights:
            weights = np.ones(len(coordinates))
        else:
            weights = np.array(
                [self.parameters.residue_weights.get(res.name, 0.0) for res in residues]
            )

        return utils.calculate_inertia_projection(coordinates, weights)

    def _apply_cached_projection(
        self, chain, cached_projection: ProjectionMatrix
    ) -> np.ndarray:
        """Apply cached projection to chain coordinates."""
        return utils.apply_projection(chain.coordinates, cached_projection)
