# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FlatProt CLI commands."""

import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

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
    """Create a temporary TOML file with valid content for testing."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        # Create a valid TOML file with all required sections
        f.write(
            b"""
[helix]
color = "#FF0000"
line_width = 2.0

[sheet]
color = "#00FF00"
line_width = 2.0

[point]
color = "#0000FF"
radius = 1.0

[line]
color = "#FFFF00"
stroke_width = 1.0

[area]
color = "#00FFFF"
opacity = 0.5
"""
        )
        f.flush()
        temp_path = Path(f.name)

    yield temp_path

    # Clean up the file
    os.unlink(temp_path)


def test_main_with_required_args(temp_structure_file, capfd):
    """Test the main function with only required arguments."""
    output_file = "test_output.svg"

    # Mock the get_coordinate_manager function to avoid actual transformation
    with (
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_transform,
        patch("flatprot.cli.commands.GemmiStructureParser") as mock_parser,
        patch("flatprot.cli.commands.generate_svg") as mock_generate_svg,
        patch("flatprot.cli.commands.save_svg") as mock_save_svg,
    ):
        # Set up mock parser to return a mock structure
        mock_structure = MagicMock()
        # Mock chain properties
        mock_chain = MagicMock()
        mock_chain.id = "A"
        mock_chain.num_residues = 10
        # Mock secondary structure
        mock_helix = MagicMock()
        mock_helix.start = 0
        mock_helix.end = 9
        mock_chain.secondary_structure = [mock_helix]
        # Make structure iterable to return the chain
        mock_structure.__iter__.return_value = [mock_chain]

        mock_parser.return_value.parse_structure.return_value = mock_structure

        # Return a mock coordinate manager
        mock_transform.return_value = MagicMock(spec=CoordinateManager)

        # Mock the generate_svg function to return a valid SVG
        mock_generate_svg.return_value = "<svg>Test SVG</svg>"

        result = app([temp_structure_file, output_file])

        assert result == 0

        # Verify interactions
        mock_parser.return_value.parse_structure.assert_called_once()
        mock_transform.assert_called_once()
        mock_generate_svg.assert_called_once()
        mock_save_svg.assert_called_once_with("<svg>Test SVG</svg>", Path(output_file))

        # Check output messages
        captured = capfd.readouterr()
        assert "Successfully processed structure" in captured.out


def test_main_with_all_args(
    temp_structure_file, temp_matrix_file, temp_toml_file, capfd
):
    """Test the main function with all arguments."""
    output_file = "test_output.svg"

    # Mock the get_coordinate_manager function to avoid actual transformation
    with (
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_transform,
        patch("flatprot.cli.commands.GemmiStructureParser") as mock_parser,
        patch("flatprot.cli.commands.generate_svg") as mock_generate_svg,
        patch("flatprot.cli.commands.save_svg") as mock_save_svg,
    ):
        # Set up mock parser to return a mock structure
        mock_structure = MagicMock()
        # Mock chain properties
        mock_chain = MagicMock()
        mock_chain.id = "A"
        mock_chain.num_residues = 10
        # Mock secondary structure
        mock_helix = MagicMock()
        mock_helix.start = 0
        mock_helix.end = 9
        mock_chain.secondary_structure = [mock_helix]
        # Make structure iterable to return the chain
        mock_structure.__iter__.return_value = [mock_chain]

        mock_parser.return_value.parse_structure.return_value = mock_structure

        # Return a mock coordinate manager
        mock_transform.return_value = MagicMock(spec=CoordinateManager)

        # Mock the generate_svg function to return a valid SVG
        mock_generate_svg.return_value = "<svg>Test SVG</svg>"

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

        # Verify interactions
        mock_parser.return_value.parse_structure.assert_called_once()
        mock_transform.assert_called_once_with(mock_structure, Path(temp_matrix_file))
        mock_generate_svg.assert_called_once_with(
            mock_structure,
            mock_transform.return_value,
            Path(temp_toml_file),
            Path(temp_toml_file),
        )
        mock_save_svg.assert_called_once_with("<svg>Test SVG</svg>", Path(output_file))

        # Check output messages
        captured = capfd.readouterr()
        assert "Successfully processed structure" in captured.out
        assert "Custom matrix" in captured.out


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

        # Mock the required functions
        with (
            patch("flatprot.cli.commands.get_coordinate_manager") as mock_transform,
            patch("flatprot.cli.commands.GemmiStructureParser") as mock_parser,
            patch("flatprot.cli.commands.generate_svg") as mock_generate_svg,
            patch("flatprot.cli.commands.save_svg") as mock_save_svg,
            patch("os.makedirs") as mock_makedirs,
        ):
            # Set up mock parser to return a mock structure
            mock_structure = MagicMock()
            # Mock chain properties
            mock_chain = MagicMock()
            mock_chain.id = "A"
            mock_chain.num_residues = 10
            # Mock secondary structure
            mock_helix = MagicMock()
            mock_helix.start = 0
            mock_helix.end = 9
            mock_chain.secondary_structure = [mock_helix]
            # Make structure iterable to return the chain
            mock_structure.__iter__.return_value = [mock_chain]

            mock_parser.return_value.parse_structure.return_value = mock_structure

            # Return a mock coordinate manager
            mock_transform.return_value = MagicMock(spec=CoordinateManager)

            # Mock the generate_svg function to return a valid SVG
            mock_generate_svg.return_value = "<svg>Test SVG</svg>"

            output_path = "nonexistent_dir/output.svg"
            result = app([f.name, output_path])

            assert result == 0

            # Verify the directory creation was attempted
            mock_makedirs.assert_called()
            mock_save_svg.assert_called_once()
