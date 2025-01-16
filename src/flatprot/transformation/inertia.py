# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .base import Transformer, TransformParameters
from .utils import (
    TransformationMatrix,
    calculate_inertia_transformation,
    apply_transformation,
)
from flatprot.structure.residue import Residue


@dataclass
class InertiaTransformerParameters:
    """Parameters for inertia-based projection calculation."""

    residue_weights: dict[Residue, float]  # Maps residue type to weight
    use_weights: bool = True

    @classmethod
    def default(cls) -> "InertiaTransformerParameters":
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
class InertiaTransformParameters(TransformParameters):
    """Parameters for inertia-based transformation calculation."""

    residues: list[Residue]


class InertiaTransformer(Transformer):
    """Transforms using inertia-based calculation with residue weights."""

    def __init__(self, parameters: Optional[InertiaTransformerParameters] = None):
        super().__init__()
        self.parameters = parameters or InertiaTransformerParameters.default()

    def _calculate_transformation(
        self,
        coordinates: np.ndarray,
        parameters: Optional[InertiaTransformParameters] = None,
    ) -> TransformationMatrix:
        """Calculate transformation matrix for given coordinates."""
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

        return calculate_inertia_transformation(coordinates, weights)

    def _apply_cached_transformation(
        self,
        coordinates: np.ndarray,
        cached_transformation: TransformationMatrix,
        parameters: Optional[InertiaTransformParameters] = None,
    ) -> np.ndarray:
        """Apply cached transformation to coordinates."""
        return apply_transformation(coordinates, cached_transformation)
