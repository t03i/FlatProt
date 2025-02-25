# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FlatProt CLI commands."""

import os
import tempfile

import pytest


from flatprot.cli.main import app
from flatprot.cli.errors import FlatProtCLIError


@pytest.fixture
def temp_structure_file():
    """Create a temporary PDB file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(
            b"ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00      A    C  \n"
        )
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def temp_matrix_file():
    """Create a temporary numpy matrix file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def temp_toml_file():
    """Create a temporary TOML file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


def test_main_with_required_args(temp_structure_file, capfd):
    """Test the main function with only required arguments."""
    output_file = "test_output.svg"

    result = app([temp_structure_file, output_file])

    assert result == 0
    captured = capfd.readouterr()
    assert "Successfully parsed arguments" in captured.out
    assert f"Structure file: {temp_structure_file}" in captured.out
    assert f"Output file: {output_file}" in captured.out


def test_main_with_all_args(
    temp_structure_file, temp_matrix_file, temp_toml_file, capfd
):
    """Test the main function with all arguments."""
    output_file = "test_output.svg"

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
    assert "Successfully parsed arguments" in captured.out
    assert f"Structure file: {temp_structure_file}" in captured.out
    assert f"Output file: {output_file}" in captured.out
    assert f"Matrix file: {temp_matrix_file}" in captured.out
    assert f"Annotations file: {temp_toml_file}" in captured.out
    assert f"Style file: {temp_toml_file}" in captured.out


def test_main_with_nonexistent_structure_file():
    """Test the main function with a nonexistent structure file."""
    with pytest.raises(FlatProtCLIError):
        app(["nonexistent_file.pdb", "output.svg"])


def test_main_with_nonexistent_output_directory():
    """Test the main function with a nonexistent output directory."""
    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        # This should not raise an error as the command should create the directory
        result = app([f.name, "nonexistent_dir/output.svg"])
        assert result == 0

        # Clean up
        if os.path.exists("nonexistent_dir"):
            os.rmdir("nonexistent_dir")
