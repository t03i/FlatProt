# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Loading matrix files for custom transformations."""

from pathlib import Path
from typing import Union

import numpy as np

from ..transformation.transformation_matrix import TransformationMatrix
from .errors import (
    MatrixFileNotFoundError,
    MatrixFileError,
    InvalidMatrixFormatError,
    InvalidMatrixDimensionsError,
)


class MatrixLoader:
    """Loads transformation matrices from numpy files."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with the path to a matrix file.

        Args:
            file_path: Path to the numpy matrix file (.npy)

        Raises:
            MatrixFileNotFoundError: If the file doesn't exist
            MatrixFileError: If the file can't be read
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise MatrixFileNotFoundError(str(self.file_path))

        if not self.file_path.suffix.lower() == ".npy":
            raise MatrixFileError(f"File must be a .npy file: {self.file_path}")

    def load(self) -> TransformationMatrix:
        """Load and validate the matrix file.

        Returns:
            TransformationMatrix object

        Raises:
            InvalidMatrixFormatError: If the matrix can't be loaded
            InvalidMatrixDimensionsError: If the matrix has invalid dimensions
        """
        try:
            matrix_array = np.load(self.file_path, allow_pickle=False)
        except Exception as e:
            raise InvalidMatrixFormatError(f"Failed to load matrix file: {e}")

        # Check matrix dimensions
        if matrix_array.ndim != 2:
            raise InvalidMatrixDimensionsError(
                f"Matrix must be 2-dimensional, got {matrix_array.ndim} dimensions"
            )

        # TransformationMatrix expects a 4x3 matrix where:
        # - First 3 rows are the rotation matrix (3x3)
        # - Last row is the translation vector (1x3)
        if matrix_array.shape == (4, 3):
            # Matrix is already in the expected format (4x3)
            try:
                return TransformationMatrix.from_array(matrix_array)
            except ValueError as e:
                raise InvalidMatrixDimensionsError(f"Invalid matrix format: {e}")
        elif matrix_array.shape == (3, 4):
            # Matrix is transposed (3x4), try to transpose it
            try:
                return TransformationMatrix.from_array(matrix_array.T)
            except ValueError as e:
                raise InvalidMatrixDimensionsError(f"Invalid matrix format: {e}")
        elif matrix_array.shape == (3, 3):
            # Only rotation matrix provided, assume zero translation
            try:
                # Create a 4x3 matrix with zero translation
                full_matrix = np.vstack([matrix_array, np.zeros(3)])
                return TransformationMatrix.from_array(full_matrix)
            except ValueError as e:
                raise InvalidMatrixDimensionsError(f"Invalid rotation matrix: {e}")
        else:
            raise InvalidMatrixDimensionsError(
                f"Matrix must have shape (4, 3), (3, 4), or (3, 3), got {matrix_array.shape}"
            )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> TransformationMatrix:
        """Convenience method to load a matrix from a file.

        Args:
            file_path: Path to the numpy matrix file

        Returns:
            TransformationMatrix object
        """
        loader = cls(file_path)
        return loader.load()
