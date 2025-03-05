# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structure file validation in the FlatProt CLI."""

import os
import tempfile
from pathlib import Path

import pytest

from flatprot.cli.commands import validate_structure_file
from flatprot.cli.errors import FileNotFoundError, InvalidStructureError


def test_validate_nonexistent_file():
    """Test validation of a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        validate_structure_file(Path("nonexistent_file.pdb"))


def test_validate_invalid_extension():
    """Test validation of a file with an invalid extension."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        with pytest.raises(InvalidStructureError):
            validate_structure_file(Path(f.name))


def test_validate_invalid_pdb_content():
    """Test validation of a file with invalid PDB content."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(b"This is not a valid PDB file")
        temp_file = f.name

    try:
        with pytest.raises(InvalidStructureError):
            validate_structure_file(Path(temp_file))
    finally:
        os.unlink(temp_file)


def test_validate_invalid_cif_content():
    """Test validation of a file with invalid CIF content."""
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        f.write(b"This is not a valid CIF file")
        temp_file = f.name

    try:
        with pytest.raises(InvalidStructureError):
            validate_structure_file(Path(temp_file))
    finally:
        os.unlink(temp_file)


def test_validate_valid_pdb_content():
    """Test validation of a file with valid PDB content."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(
            b"ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00      A    C  \n"
        )
        temp_file = f.name

    try:
        # This should not raise an exception
        validate_structure_file(Path(temp_file))
    finally:
        os.unlink(temp_file)


def test_validate_valid_cif_content():
    """Test validation of a file with valid CIF content."""
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        f.write(b"data_test\nloop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n")
        temp_file = f.name

    try:
        # This should not raise an exception
        validate_structure_file(Path(temp_file))
    finally:
        os.unlink(temp_file)


def test_validate_binary_file():
    """Test validation of a binary file."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(bytes([0, 1, 2, 3, 4, 5]))  # Non-text binary content
        temp_file = f.name

    try:
        with pytest.raises(InvalidStructureError):
            validate_structure_file(Path(temp_file))
    finally:
        os.unlink(temp_file)
