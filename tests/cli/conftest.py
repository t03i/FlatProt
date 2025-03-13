# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for the FlatProt CLI tests."""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
from unittest.mock import MagicMock

from flatprot.core import (
    Structure,
    Chain,
    SecondaryStructure,
    SecondaryStructureType,
    CoordinateManager,
    CoordinateType,
    Residue,
)


@pytest.fixture
def mock_coordinate_manager():
    """Create a mock coordinate manager for testing."""
    cm = CoordinateManager()

    # Mock original coordinates
    original_coords = np.array(
        [
            [1.0, 2.0, 3.0],  # 0
            [2.0, 3.0, 4.0],  # 1
            [3.0, 4.0, 5.0],  # 2
            [4.0, 5.0, 6.0],  # 3
            [5.0, 6.0, 7.0],  # 4
            [6.0, 7.0, 8.0],  # 5
        ]
    )

    # Mock transformed coordinates
    transformed_coords = np.array(
        [
            [2.0, 3.0, 4.0],  # 0
            [3.0, 4.0, 5.0],  # 1
            [4.0, 5.0, 6.0],  # 2
            [5.0, 6.0, 7.0],  # 3
            [6.0, 7.0, 8.0],  # 4
            [7.0, 8.0, 9.0],  # 5
        ]
    )

    # Mock canvas coordinates
    canvas_coords = np.array(
        [
            [100.0, 200.0],  # 0
            [150.0, 250.0],  # 1
            [200.0, 300.0],  # 2
            [250.0, 350.0],  # 3
            [300.0, 400.0],  # 4
            [350.0, 450.0],  # 5
        ]
    )

    # Mock depth values
    depth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # Add all coordinates to the manager
    cm.add(0, 6, original_coords, CoordinateType.COORDINATES)
    cm.add(0, 6, transformed_coords, CoordinateType.TRANSFORMED)
    cm.add(0, 6, canvas_coords, CoordinateType.CANVAS)
    cm.add(0, 6, depth, CoordinateType.DEPTH)

    return cm


@pytest.fixture
def temp_annotation_file():
    """Create a temporary annotation file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        content = """
        [[annotations]]
        type = "point"
        label = "Residue 1"
        index = 1
        chain = "A"

        [[annotations]]
        type = "line"
        label = "Connection 1-3"
        indices = [1, 3]
        chain = "A"

        [[annotations]]
        type = "area"
        label = "Region 1-5"
        range = { start = 1, end = 5 }
        chain = "A"
        """
        tmp.write(content.encode())
        tmp.flush()
        path = Path(tmp.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_style_file():
    """Create a temporary style file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        content = """
        [helix]
        color = "#FF0000"
        line_width = 2.0

        [sheet]
        color = "#00FF00"
        line_width = 3.0
        min_sheet_length = 2

        [coil]
        color = "#0000FF"
        line_width = 1.0
        """
        tmp.write(content.encode())
        tmp.flush()
        path = Path(tmp.name)
    yield path
    path.unlink(missing_ok=True)


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


@pytest.fixture
def mock_structure():
    """Create a mock structure for testing."""
    structure = MagicMock(spec=Structure)

    # Create a mock chain
    chain = MagicMock(spec=Chain)
    chain.id = "A"

    # Create mock secondary structure elements
    helix = MagicMock(spec=SecondaryStructure)
    helix.start = 0
    helix.end = 2
    helix.secondary_structure_type = SecondaryStructureType.HELIX

    sheet = MagicMock(spec=SecondaryStructure)
    sheet.start = 3
    sheet.end = 5
    sheet.secondary_structure_type = SecondaryStructureType.SHEET

    # Add secondary structure elements to chain
    chain.secondary_structure = [helix, sheet]

    # Mock len() for chain
    chain.num_residues = 6

    # Add chain to structure
    structure.__iter__.return_value = [chain]

    # Mock coordinates
    structure.coordinates = np.array(
        [
            [1.0, 2.0, 3.0],  # 0
            [2.0, 3.0, 4.0],  # 1
            [3.0, 4.0, 5.0],  # 2
            [4.0, 5.0, 6.0],  # 3
            [5.0, 6.0, 7.0],  # 4
            [6.0, 7.0, 8.0],  # 5
        ]
    )
    structure.residues = [
        Residue.ALA,
        Residue.ALA,
        Residue.ALA,
        Residue.ALA,
        Residue.ALA,
        Residue.ALA,
    ]

    return structure


@pytest.fixture
def valid_matrix_file():
    """Create a temporary file with a valid transformation matrix."""
    # Create a valid transformation matrix (4x3)
    rotation = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    matrix = np.vstack([rotation, translation])

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, matrix)

    yield Path(path)

    # Clean up
    os.unlink(path)


@pytest.fixture
def invalid_matrix_file():
    """Create a temporary file with an invalid matrix."""
    # Create an invalid matrix (wrong dimensions)
    matrix = np.eye(3)

    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, matrix)

    yield Path(path)

    # Clean up
    os.unlink(path)
