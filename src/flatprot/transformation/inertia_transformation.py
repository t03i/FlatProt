# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import numpy as np

from .base_transformation import (
    BaseTransformation,
    BaseTransformationArguments,
    BaseTransformationParameters,
)
from .transformation_matrix import (
    TransformationMatrix,
)
from flatprot.core.types import ResidueType


def calculate_inertia_transformation_matrix(
    coordinates: np.ndarray, weights: np.ndarray, use_lsq: bool = True
) -> TransformationMatrix:
    """Calculate transformation matrix for optimal molecular orientation.

    Implements the X-PLOR/CNS orient command algorithm:
    1. Translates geometric center/center of mass to origin
    2. Determines rotation either via Kabsch algorithm

    The transformation follows r' = R*r + T where:
    - r' is the oriented coordinate set
    - R is the rotation matrix
    - r is the original coordinates
    - T is the translation vector

    Args:
        coordinates: Nx3 array of atomic coordinates
        weights: N-length array of weights for each coordinate
    """
    # Calculate center (weighted or geometric)
    if np.allclose(weights, weights[0]):  # All weights equal -> geometric center
        translation = np.mean(coordinates, axis=0)
    else:  # Weighted center of mass
        total_weight = np.sum(weights)
        translation = (
            np.sum(coordinates * weights[:, np.newaxis], axis=0) / total_weight
        )

    # Center coordinates
    centered = coordinates - translation

    # Principal axis method using inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for coord, weight in zip(centered, weights):
        r_squared = np.sum(coord * coord)
        inertia_tensor += weight * (r_squared * np.eye(3) - np.outer(coord, coord))

    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    rotation = eigenvectors

    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1

    return TransformationMatrix(rotation=rotation, translation=translation)


@dataclass
class InertiaTransformationParameters(BaseTransformationParameters):
    """Parameters for inertia-based projection calculation."""

    residue_weights: dict[ResidueType, float]  # Maps residue type to weight
    use_weights: bool = True

    @classmethod
    def default(cls) -> "InertiaTransformationParameters":
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
class InertiaTransformationArguments(BaseTransformationArguments):
    """Arguments for inertia-based transformation."""

    residues: list[ResidueType]


class InertiaTransformer(
    BaseTransformation[InertiaTransformationParameters, InertiaTransformationArguments]
):
    """Transforms using inertia-based calculation with residue weights."""

    def __init__(self, parameters: InertiaTransformationParameters):
        super().__init__(parameters=parameters)

    def _calculate_transformation_matrix(
        self,
        coordinates: np.ndarray,
        parameters: InertiaTransformationParameters,
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

        return calculate_inertia_transformation_matrix(coordinates, weights)
