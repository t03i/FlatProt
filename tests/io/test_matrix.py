# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for matrix loading functionality."""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

from flatprot.io.matrix import MatrixLoader
from flatprot.transformation.transformation_matrix import TransformationMatrix
from flatprot.io import (
    InvalidMatrixDimensionsError,
    InvalidMatrixFormatError,
    MatrixFileError,
    MatrixFileNotFoundError,
)


def create_valid_matrix_file() -> Path:
    """Create a temporary file with a valid transformation matrix."""
    # Create a valid transformation matrix (4x3)
    rotation = np.eye(3)  # Identity rotation
    translation = np.array([1.0, 2.0, 3.0])  # Simple translation
    matrix = np.vstack([rotation, translation])  # Combined 4x3 matrix

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, matrix)
    return Path(path)


def create_transposed_matrix_file() -> Path:
    """Create a temporary file with a transposed (3x4) transformation matrix."""
    # Create a valid transformation matrix but transposed (3x4)
    rotation = np.eye(3)  # Identity rotation
    translation = np.array([1.0, 2.0, 3.0])  # Simple translation
    matrix = np.vstack([rotation, translation])  # Combined 4x3 matrix
    matrix = matrix.T  # Transpose to 3x4

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, matrix)
    return Path(path)


def create_invalid_dimensions_file() -> Path:
    """Create a temporary file with an invalid matrix shape."""
    # Create a 2x2 matrix (invalid for transformation)
    matrix = np.eye(2)  # 2x2 identity matrix

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, matrix)
    return Path(path)


def create_invalid_format_file() -> Path:
    """Create a temporary file with invalid content (not a matrix)."""
    # Create some random non-matrix data
    data = np.array([1, 2, 3, 4, 5])

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, data)
    return Path(path)


def create_corrupted_file() -> Path:
    """Create a temporary file with corrupted content."""
    # Create a corrupted file
    fd, path = tempfile.mkstemp(suffix=".npy")
    with os.fdopen(fd, "w") as f:
        f.write("This is not a valid numpy file")
    return Path(path)


def create_rotation_only_matrix_file() -> Path:
    """Create a temporary file with only a rotation matrix (3x3)."""
    # Create a rotation matrix (3x3)
    rotation = np.eye(3)  # Identity rotation

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, rotation)
    return Path(path)


def test_valid_matrix_load():
    """Test loading a valid transformation matrix."""
    file_path = create_valid_matrix_file()
    try:
        loader = MatrixLoader(file_path)
        matrix = loader.load()

        # Verify the matrix is loaded correctly
        assert isinstance(matrix, TransformationMatrix)
        assert np.array_equal(matrix.rotation, np.eye(3))
        assert np.array_equal(matrix.translation, np.array([1.0, 2.0, 3.0]))
    finally:
        os.unlink(file_path)


def test_transposed_matrix_load():
    """Test loading a transposed (3x4) transformation matrix."""
    file_path = create_transposed_matrix_file()
    try:
        loader = MatrixLoader(file_path)
        matrix = loader.load()

        # Verify the matrix is loaded and transposed correctly
        assert isinstance(matrix, TransformationMatrix)
        assert np.array_equal(matrix.rotation, np.eye(3))
        assert np.array_equal(matrix.translation, np.array([1.0, 2.0, 3.0]))
    finally:
        os.unlink(file_path)


def test_invalid_dimensions():
    """Test loading a matrix with invalid dimensions."""
    file_path = create_invalid_dimensions_file()
    try:
        loader = MatrixLoader(file_path)
        with pytest.raises(InvalidMatrixDimensionsError) as excinfo:
            loader.load()
        assert "Matrix must have shape (4, 3), (3, 4), or (3, 3)" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_invalid_format():
    """Test loading a file with invalid format (not a 2D matrix)."""
    file_path = create_invalid_format_file()
    try:
        loader = MatrixLoader(file_path)
        with pytest.raises(InvalidMatrixDimensionsError) as excinfo:
            loader.load()
        assert "Matrix must be 2-dimensional" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_corrupted_file():
    """Test loading a corrupted file."""
    file_path = create_corrupted_file()
    try:
        loader = MatrixLoader(file_path)
        with pytest.raises(InvalidMatrixFormatError) as excinfo:
            loader.load()
        assert "Failed to load matrix file" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(MatrixFileNotFoundError) as excinfo:
        MatrixLoader("non_existent_file.npy")
    assert "Matrix file not found" in str(excinfo.value)


def test_wrong_file_extension():
    """Test handling of files with wrong extension."""
    # Create temporary file with wrong extension
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with pytest.raises(MatrixFileError) as excinfo:
            MatrixLoader(path)
        assert "File must be a .npy file" in str(excinfo.value)
    finally:
        os.unlink(path)


def test_from_file_convenience_method():
    """Test the convenience method for loading matrices."""
    file_path = create_valid_matrix_file()
    try:
        # Use the class method directly
        matrix = MatrixLoader.from_file(file_path)

        # Verify the matrix is loaded correctly
        assert isinstance(matrix, TransformationMatrix)
        assert np.array_equal(matrix.rotation, np.eye(3))
        assert np.array_equal(matrix.translation, np.array([1.0, 2.0, 3.0]))
    finally:
        os.unlink(file_path)


def test_rotation_only_matrix_load():
    """Test loading a matrix with only rotation (3x3)."""
    file_path = create_rotation_only_matrix_file()
    try:
        loader = MatrixLoader(file_path)
        matrix = loader.load()

        # Verify the matrix is loaded correctly
        assert isinstance(matrix, TransformationMatrix)
        assert np.array_equal(matrix.rotation, np.eye(3))
        assert np.array_equal(matrix.translation, np.zeros(3))
    finally:
        os.unlink(file_path)
