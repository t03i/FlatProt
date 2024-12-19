# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .projector import Projector, ProjectionParameters
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


@dataclass
class InertiaProjectionParameters(ProjectionParameters):
    """Parameters for inertia-based projection calculation."""

    residues: list[str]


class InertiaProjector(Projector):
    """Projects using inertia-based calculation with residue weights."""

    def __init__(self, parameters: Optional[InertiaParameters] = None):
        super().__init__()
        self.parameters = parameters or InertiaParameters.default()

    def _calculate_projection(
        self,
        coordinates: np.ndarray,
        parameters: Optional[InertiaProjectionParameters] = None,
    ) -> ProjectionMatrix:
        """Calculate projection matrix for given coordinates."""
        if not self.parameters.use_weights:
            weights = np.ones(len(coordinates))
        else:
            # Map residue types to weights using parameters.residue_weights
            weights = np.array(
                [
                    self.parameters.residue_weights.get(res, 1.0)
                    for res in parameters.residues
                ]
            )

        return utils.calculate_inertia_projection(coordinates, weights)

    def _apply_cached_projection(
        self,
        coordinates: np.ndarray,
        cached_projection: ProjectionMatrix,
        parameters: Optional[InertiaProjectionParameters] = None,
    ) -> np.ndarray:
        """Apply cached projection to coordinates."""
        return utils.apply_projection(coordinates, cached_projection)
