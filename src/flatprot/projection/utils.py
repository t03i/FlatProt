# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectionMatrix:
    """Represents a projection transformation."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        """Validate matrix shapes."""
        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if self.translation.shape != (3,):
            raise ValueError("Translation vector must have shape (3,)")

    def combined_rotation(self, other: "ProjectionMatrix") -> "ProjectionMatrix":
        """Combine two projection matrices by first applying self, then other.

        The combined transformation T = T2 ∘ T1 is:
        rotation = R2 @ R1
        translation = R2 @ t1 + t2
        """
        return ProjectionMatrix(
            rotation=other.rotation @ self.rotation,
            translation=other.rotation @ self.translation + other.translation,
        )

    def to_array(self) -> np.ndarray:
        """Convert ProjectionMatrix to a single numpy array for storage.

        Returns:
            Array of shape (4, 3) where first 3 rows are rotation matrix
            and last row is translation vector
        """
        return np.vstack([self.rotation, self.translation])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ProjectionMatrix":
        """Create ProjectionMatrix from stored array format.

        Args:
            arr: Array of shape (4, 3) where first 3 rows are rotation matrix
                and last row is translation vector

        Returns:
            ProjectionMatrix instance
        """
        return cls(rotation=arr[0:3, :], translation=arr[3, :])

    @classmethod
    def from_string(cls, s: str) -> "ProjectionMatrix":
        """Create ProjectionMatrix from string representation."""
        arr = np.fromstring(s)
        arr = arr.reshape(4, 3)
        return cls.from_array(arr)


def calculate_inertia_projection(
    coordinates: np.ndarray, weights: np.ndarray
) -> ProjectionMatrix:
    """Calculate projection matrix using weighted inertia tensor.

    Args:
        coordinates: Nx3 array of atomic coordinates
        weights: N-length array of weights for each coordinate
    """
    # Calculate center of mass
    total_weight = np.sum(weights)
    com = np.sum(coordinates * weights[:, np.newaxis], axis=0) / total_weight

    # Center coordinates
    centered_coords = coordinates - com

    # Calculate weighted moment of inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for coord, weight in zip(centered_coords, weights):
        r_squared = np.sum(coord * coord)
        inertia_tensor += weight * (r_squared * np.eye(3) - np.outer(coord, coord))

    # Get eigenvectors and sort by eigenvalue magnitude
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    idx = np.argsort(np.abs(eigenvalues))
    rotation = eigenvectors[:, idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    return ProjectionMatrix(rotation=rotation, translation=com)


def apply_projection(
    coordinates: np.ndarray, projection: ProjectionMatrix
) -> np.ndarray:
    """Apply projection matrix to coordinates.

    Args:
        coordinates: Array of shape (N, 3) containing 3D coordinates
        projection: ProjectionMatrix containing rotation and translation

    Returns:
        Array of shape (N, 3) containing transformed coordinates
    """
    # First center by subtracting translation
    centered = coordinates - projection.translation
    # Then apply rotation
    rotated = centered @ projection.rotation.T
    return rotated
