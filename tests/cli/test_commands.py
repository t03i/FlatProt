# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FlatProt CLI commands."""

import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

import pytest

from flatprot.cli.main import app
from flatprot.core import CoordinateManager


@pytest.fixture
def temp_structure_file():
    """Create a temporary PDB file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        # Add PDB content with ATOM records
        f.write(
            b"HEADER    PROTEIN\n"
            b"ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00      A    C  \n"
            b"ATOM      2  CA  ALA A   2      10.000  10.000  11.000  1.00 20.00      A    C  \n"
            b"ATOM      3  CA  ALA A   3      10.000  11.000  11.000  1.00 20.00      A    C  \n"
            b"END\n"
        )
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def temp_matrix_file():
    """Create a temporary numpy matrix file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        # Create a valid transformation matrix (4x3)
        rotation = np.eye(3)  # Identity rotation
        translation = np.array([1.0, 2.0, 3.0])  # Simple translation
        matrix = np.vstack([rotation, translation])  # Combined 4x3 matrix

        temp_file = f.name
        # Save the matrix to the temporary file
        np.save(temp_file, matrix)
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def temp_toml_file():
    """Create a temporary TOML file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        # Add simple TOML content
        f.write(b'[section]\nkey = "value"\n')
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


def test_main_with_required_args(temp_structure_file, capfd):
    """Test the main function with only required arguments."""
    output_file = "test_output.svg"

    # Mock the get_coordinate_manager function to avoid actual transformation
    with patch("flatprot.cli.commands.get_coordinate_manager") as mock_transform:
        # Return a mock coordinate manager
        mock_transform.return_value = MagicMock(spec=CoordinateManager)

        result = app([temp_structure_file, output_file])

        assert result == 0
        captured = capfd.readouterr()
        assert "Successfully processed structure" in captured.out
        # Use 'in' operator for more flexible path matching due to platform differences
        assert str(temp_structure_file) in captured.out
        assert output_file in captured.out


def test_main_with_all_args(
    temp_structure_file, temp_matrix_file, temp_toml_file, capfd
):
    """Test the main function with all arguments."""
    output_file = "test_output.svg"

    # Mock the get_coordinate_manager function to avoid actual transformation
    with patch("flatprot.cli.commands.get_coordinate_manager") as mock_transform:
        # Return a mock coordinate manager
        mock_transform.return_value = MagicMock(spec=CoordinateManager)

        result = app(
            [
                str(temp_structure_file),
                output_file,
                "--matrix",
                str(temp_matrix_file),
                "--annotations",
                str(temp_toml_file),
                "--style",
                str(temp_toml_file),
            ]
        )

        assert result == 0
        captured = capfd.readouterr()
        assert "Successfully processed structure" in captured.out
        # Use 'in' operator for more flexible path matching
        assert str(temp_structure_file) in captured.out
        assert output_file in captured.out
        assert str(temp_matrix_file) in captured.out
        assert str(temp_toml_file) in captured.out


def test_main_with_nonexistent_structure_file(capfd):
    """Test the main function with a nonexistent structure file."""
    result = app(["nonexistent_file.pdb", "output.svg"])
    assert result == 1  # Should return error code
    captured = capfd.readouterr()
    assert "Error:" in captured.out
    assert "not found" in captured.out


def test_main_with_nonexistent_output_directory():
    """Test the main function with a nonexistent output directory."""
    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        # Write valid content to the temp file
        f.write(
            b"HEADER    PROTEIN\n"
            b"ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00      A    C  \n"
            b"END\n"
        )
        f.flush()  # Ensure content is written before testing

        # Mock the get_coordinate_manager function to avoid actual transformation
        with patch("flatprot.cli.commands.get_coordinate_manager") as mock_transform:
            # Return a mock coordinate manager
            mock_transform.return_value = MagicMock(spec=CoordinateManager)

            output_path = "nonexistent_dir/output.svg"
            result = app([f.name, output_path])
            assert result == 0

            # Clean up
            if os.path.exists("nonexistent_dir"):
                os.rmdir("nonexistent_dir")
