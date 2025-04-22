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

    def before(self, other: "TransformationMatrix") -> "TransformationMatrix":
        """Combine two transformation matrices by first applying self, then other.

        The combined transformation T = T2 ∘ T1 is:
        rotation = R2 @ R1
        translation = R2 @ t1 + t2
        """
        return TransformationMatrix(
            rotation=other.rotation @ self.rotation,
            translation=other.rotation @ self.translation + other.translation,
        )

    def after(self, other: "TransformationMatrix") -> "TransformationMatrix":
        """Combine two transformation matrices by first applying other, then self.

        The combined transformation T = T1 ∘ T2 is:
        rotation = R1 @ R2
        translation = R1 @ t2 + t1
        """
        return TransformationMatrix(
            rotation=self.rotation @ other.rotation,
            translation=self.rotation @ other.translation + self.translation,
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
        """
        Apply the transformation matrix using the standard (R @ X) + T convention.

        Args:
            coordinates: Array of shape (N, 3) containing 3D coordinates.

        Returns:
            Array of shape (N, 3) containing transformed coordinates.
        """
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("Input coordinates must be a numpy array.")
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                f"Input coordinates must have shape (N, 3), got {coordinates.shape}"
            )
        if coordinates.size == 0:
            return coordinates  # Return empty array if input is empty

        # Standard application: Rotate around origin, then translate
        rotated = (self.rotation @ coordinates.T).T
        transformed = rotated + self.translation
        return transformed

    def __eq__(self, other: "TransformationMatrix") -> bool:
        """Check if two TransformationMatrix instances are equal."""
        return np.allclose(self.rotation, other.rotation) and np.allclose(
            self.translation, other.translation
        )
