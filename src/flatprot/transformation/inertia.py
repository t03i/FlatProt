# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np
from typing import Optional

from .base import BaseTransformer, TransformParameters
from .utils import (
    TransformationMatrix,
    calculate_inertia_transformation,
    apply_transformation,
)
from flatprot.core.types import ResidueType


@dataclass
class InertiaTransformerParameters:
    """Parameters for inertia-based projection calculation."""

    residue_weights: dict[ResidueType, float]  # Maps residue type to weight
    use_weights: bool = True

    @classmethod
    def default(cls) -> "InertiaTransformerParameters":
        """Creates default parameters using standard amino acid weights."""
        return cls(
            residue_weights={
                ResidueType.ALA: 89.1,
                ResidueType.ARG: 174.2,
                ResidueType.ASN: 132.1,
                ResidueType.ASP: 133.1,
                ResidueType.CYS: 121.2,
                ResidueType.GLN: 146.2,
                ResidueType.GLU: 147.1,
                ResidueType.GLY: 75.1,
                ResidueType.HIS: 155.2,
                ResidueType.ILE: 131.2,
                ResidueType.LEU: 131.2,
                ResidueType.LYS: 146.2,
                ResidueType.MET: 149.2,
                ResidueType.PHE: 165.2,
                ResidueType.PRO: 115.1,
                ResidueType.SER: 105.1,
                ResidueType.THR: 119.1,
                ResidueType.TRP: 204.2,
                ResidueType.TYR: 181.2,
                ResidueType.VAL: 117.1,
            }
        )


@dataclass
class InertiaTransformParameters(TransformParameters):
    """Parameters for inertia-based transformation calculation."""

    residues: list[ResidueType]


class InertiaTransformer(BaseTransformer):
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
