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
    coordinates: np.ndarray, weights: np.ndarray
) -> TransformationMatrix:
    """
    Calculate transformation matrix for optimal molecular orientation,
    returning components compatible with standard (R @ X) + T application.

    The transformation aligns the principal axes with the coordinate axes
    and moves the center of mass/geometry C to the origin.
    This function calculates R_inertia and C, then returns R_standard = R_inertia
    and T_standard = -(R_inertia @ C).

    Args:
        coordinates: Nx3 array of atomic coordinates
        weights: N-length array of weights for each coordinate

    Returns:
        TransformationMatrix with rotation = R_inertia and translation = -(R_inertia @ C)
    """
    # Calculate center (weighted or geometric) C
    if np.allclose(weights, weights[0]):  # All weights equal -> geometric center
        center_C = np.mean(coordinates, axis=0)
    else:  # Weighted center of mass
        total_weight = np.sum(weights)
        if total_weight == 0:
            center_C = np.mean(coordinates, axis=0)  # Fallback for zero weights
        else:
            center_C = (
                np.sum(coordinates * weights[:, np.newaxis], axis=0) / total_weight
            )

    # Center coordinates for inertia tensor calculation
    centered_coords = coordinates - center_C

    # Principal axis method using inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for coord, weight in zip(centered_coords, weights):
        r_squared = np.sum(coord * coord)
        inertia_tensor += weight * (r_squared * np.eye(3) - np.outer(coord, coord))

    _, eigenvectors = np.linalg.eigh(inertia_tensor)
    # Rotation matrix (R_inertia) is composed of the eigenvectors
    rotation_R_inertia = eigenvectors

    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation_R_inertia) < 0:
        rotation_R_inertia[:, 2] *= -1

    # Calculate the standard translation T = -(R_inertia @ C)
    # Ensure C is treated as a column vector for matmul
    center_C_col = center_C.reshape(-1, 1)
    translation_T_standard_col = -(rotation_R_inertia @ center_C_col)
    translation_T_standard = translation_T_standard_col.flatten()  # Back to (3,)

    # Return the matrix with components for standard application
    return TransformationMatrix(
        rotation=rotation_R_inertia, translation=translation_T_standard
    )


@dataclass
class InertiaTransformationParameters(BaseTransformationParameters):
    """Parameters for inertia-based projection calculation."""

    residue_weights: dict[ResidueType, float]  # Maps residue type to weight
    use_weights: bool = False

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
