# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .projector import Projector, ProjectionParameters
from .utils import TransformationMatrix
import flatprot.projection.utils as utils
from flatprot.structure.residue import Residue


@dataclass
class InertiaParameters:
    """Parameters for inertia-based projection calculation."""

    residue_weights: dict[Residue, float]  # Maps residue type to weight
    use_weights: bool = True

    @classmethod
    def default(cls) -> "InertiaParameters":
        """Creates default parameters using standard amino acid weights."""
        return cls(
            residue_weights={
                Residue.ALA: 89.1,
                Residue.ARG: 174.2,
                Residue.ASN: 132.1,
                Residue.ASP: 133.1,
                Residue.CYS: 121.2,
                Residue.GLN: 146.2,
                Residue.GLU: 147.1,
                Residue.GLY: 75.1,
                Residue.HIS: 155.2,
                Residue.ILE: 131.2,
                Residue.LEU: 131.2,
                Residue.LYS: 146.2,
                Residue.MET: 149.2,
                Residue.PHE: 165.2,
                Residue.PRO: 115.1,
                Residue.SER: 105.1,
                Residue.THR: 119.1,
                Residue.TRP: 204.2,
                Residue.TYR: 181.2,
                Residue.VAL: 117.1,
            }
        )


@dataclass
class InertiaProjectionParameters(ProjectionParameters):
    """Parameters for inertia-based projection calculation."""

    residues: list[Residue]


class InertiaProjector(Projector):
    """Projects using inertia-based calculation with residue weights."""

    def __init__(self, parameters: Optional[InertiaParameters] = None):
        super().__init__()
        self.parameters = parameters or InertiaParameters.default()

    def _calculate_projection(
        self,
        coordinates: np.ndarray,
        parameters: Optional[InertiaProjectionParameters] = None,
    ) -> TransformationMatrix:
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
        cached_projection: TransformationMatrix,
        parameters: Optional[InertiaProjectionParameters] = None,
    ) -> np.ndarray:
        """Apply cached projection to coordinates."""
        return utils.apply_projection(coordinates, cached_projection)
