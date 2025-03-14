# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structure file validation in FlatProt."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple
import pytest

from flatprot.io.structure import validate_structure_file
from flatprot.io.errors import StructureFileNotFoundError, InvalidStructureError


@pytest.fixture
def valid_pdb_file() -> Generator[str, None, None]:
    """Create a temporary file with valid PDB content.

    Yields:
        Path to the temporary PDB file
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(b"HEADER    TEST STRUCTURE\n")
        f.write(
            b"ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00      A    N\n"
        )
        f.write(
            b"ATOM      2  CA  ALA A   1      13.000  13.000  13.000  1.00 20.00      A    C\n"
        )
        f.write(b"END\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def valid_cif_file() -> Generator[str, None, None]:
    """Create a temporary file with valid mmCIF content.

    Yields:
        Path to the temporary mmCIF file
    """
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        f.write(b"data_TEST\n")
        f.write(b"loop_\n")
        f.write(b"_atom_site.group_PDB\n")
        f.write(b"_atom_site.id\n")
        f.write(b"_atom_site.type_symbol\n")
        f.write(b"_atom_site.label_atom_id\n")
        f.write(b"ATOM 1 N N\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def invalid_content_files() -> Generator[Tuple[str, str], None, None]:
    """Create temporary files with invalid content for both PDB and CIF formats.

    Yields:
        Tuple of paths to temporary invalid PDB and CIF files
    """
    with (
        tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as pdb_file,
        tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as cif_file,
    ):
        # Invalid PDB (just random text)
        pdb_file.write(b"This is not a valid PDB file content\n")
        pdb_path = pdb_file.name

        # Invalid CIF (just random text)
        cif_file.write(b"This is not a valid CIF file content\n")
        cif_path = cif_file.name

    yield (pdb_path, cif_path)

    os.unlink(pdb_path)
    os.unlink(cif_path)


@pytest.fixture
def binary_file() -> Generator[str, None, None]:
    """Create a temporary binary file that will cause UnicodeDecodeError.

    Yields:
        Path to the temporary binary file
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        # Write some binary data that will cause UnicodeDecodeError
        f.write(b"\x80\x81\x82\x83")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


def test_validate_valid_pdb_file(valid_pdb_file: str) -> None:
    """Test validation of a valid PDB file.

    Args:
        valid_pdb_file: Path to a valid PDB file
    """
    # Should not raise any exception
    validate_structure_file(Path(valid_pdb_file))


def test_validate_valid_cif_file(valid_cif_file: str) -> None:
    """Test validation of a valid mmCIF file.

    Args:
        valid_cif_file: Path to a valid mmCIF file
    """
    # Should not raise any exception
    validate_structure_file(Path(valid_cif_file))


def test_validate_nonexistent_file() -> None:
    """Test validation of a nonexistent file.

    Verifies that StructureFileNotFoundError is raised for nonexistent files.
    """
    nonexistent_path = Path("/path/to/nonexistent/file.pdb")
    with pytest.raises(StructureFileNotFoundError):
        validate_structure_file(nonexistent_path)


def test_validate_unsupported_extension() -> None:
    """Test validation of a file with an unsupported extension.

    Verifies that InvalidStructureError is raised for files with unsupported extensions.
    """
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        with pytest.raises(InvalidStructureError):
            validate_structure_file(Path(f.name))


def test_validate_invalid_pdb_content(invalid_content_files: Tuple[str, str]) -> None:
    """Test validation of a PDB file with invalid content.

    Args:
        invalid_content_files: Tuple containing paths to invalid PDB and CIF files
    """
    invalid_pdb_path = invalid_content_files[0]
    with pytest.raises(InvalidStructureError):
        validate_structure_file(Path(invalid_pdb_path))


def test_validate_invalid_cif_content(invalid_content_files: Tuple[str, str]) -> None:
    """Test validation of a CIF file with invalid content.

    Args:
        invalid_content_files: Tuple containing paths to invalid PDB and CIF files
    """
    invalid_cif_path = invalid_content_files[1]
    with pytest.raises(InvalidStructureError):
        validate_structure_file(Path(invalid_cif_path))


def test_validate_binary_file(binary_file: str) -> None:
    """Test validation of a binary file that will cause UnicodeDecodeError.

    Args:
        binary_file: Path to a binary file
    """
    with pytest.raises(InvalidStructureError):
        validate_structure_file(Path(binary_file))


def test_validate_different_extensions() -> None:
    """Test validation with various supported PDB and CIF extensions."""
    extensions = [".pdb", ".cif", ".mmcif", ".ent"]

    for ext in extensions:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            # Write minimal valid content based on extension
            if ext in [".pdb", ".ent"]:
                f.write(b"ATOM      1  N   ALA A   1      10.000  10.000  10.000\n")
            else:  # CIF formats
                f.write(b"data_TEST\n_atom_site.id")

            temp_path = f.name

        try:
            # Should not raise exception for supported extensions with minimal valid content
            validate_structure_file(Path(temp_path))
        finally:
            os.unlink(temp_path)
