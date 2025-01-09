# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np


@dataclass
class TransformationMatrix:
    """Represents a transformation matrix."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        """Validate matrix shapes."""
        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if self.translation.shape != (3,):
            raise ValueError("Translation vector must have shape (3,)")

    def combined_rotation(
        self, other: "TransformationMatrix"
    ) -> "TransformationMatrix":
        """Combine two transformation matrices by first applying self, then other.

        The combined transformation T = T2 ∘ T1 is:
        rotation = R2 @ R1
        translation = R2 @ t1 + t2
        """
        return TransformationMatrix(
            rotation=other.rotation @ self.rotation,
            translation=other.rotation @ self.translation + other.translation,
        )

    def to_array(self) -> np.ndarray:
        """Convert TransformationMatrix to a single numpy array for storage.

        Returns:
            Array of shape (4, 3) where first 3 rows are rotation matrix
            and last row is translation vector
        """
        return np.vstack([self.rotation, self.translation])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TransformationMatrix":
        """Create TransformationMatrix from stored array format.

        Args:
            arr: Array of shape (4, 3) where first 3 rows are rotation matrix
                and last row is translation vector

        Returns:
            TransformationMatrix instance
        """
        return cls(rotation=arr[0:3, :], translation=arr[3, :])

    @classmethod
    def from_string(cls, s: str) -> "TransformationMatrix":
        """Create TransformationMatrix from string representation."""
        arr = np.fromstring(s)
        arr = arr.reshape(4, 3)
        return cls.from_array(arr)


def calculate_inertia_transformation(
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


def apply_transformation(
    coordinates: np.ndarray, transformation: TransformationMatrix
) -> np.ndarray:
    """Apply transformation matrix to coordinates.

    Args:
        coordinates: Array of shape (N, 3) containing 3D coordinates
        transformation: TransformationMatrix co ntaining rotation and translation

    Returns:
        Array of shape (N, 3) containing transformed coordinates
    """
    # First center by subtracting translation
    rotated = (transformation.rotation @ coordinates.T).T
    transformed = rotated + transformation.translation
    # Then apply rotation
    return transformed
