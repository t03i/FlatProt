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
        if arr.shape != (4, 3):
            raise ValueError(f"Input array must be 4x3, but got {arr.shape}")
        # Original slicing for 4x3 array
        return cls(rotation=arr[0:3, :], translation=arr[3, :])

    @classmethod
    def from_string(cls, s: str) -> "TransformationMatrix":
        """Create TransformationMatrix from string representation (assuming 4x3 layout)."""
        # Assuming the string represents a flattened 4x3 matrix
        arr = np.fromstring(s, sep=" ")  # Adjust separator if needed
        if arr.size != 12:
            raise ValueError("String must represent 12 numbers for a 4x3 matrix")
        arr = arr.reshape(4, 3)
        return cls.from_array(arr)

    def apply(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to coordinates.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates
            transformation: TransformationMatrix co ntaining rotation and translation

        Returns:
            Array of shape (N, 3) containing transformed coordinates
        """

        assert coordinates.shape[1] == 3, "Coordinates must have shape (N, 3)"

        # First center by subtracting translation
        centered = coordinates - self.translation
        # Then apply rotation
        rotated = (self.rotation @ centered.T).T
        return rotated
