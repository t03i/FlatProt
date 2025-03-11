# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for transformation application in the CLI."""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from flatprot.cli.commands import get_coordinate_manager, main
from flatprot.cli.errors import TransformationError
from flatprot.transformation import (
    TransformationMatrix,
)
from flatprot.core import Structure, CoordinateManager, CoordinateType


@pytest.fixture
def mock_structure():
    """Create a mock structure for testing."""
    structure = MagicMock(spec=Structure)

    # Mock the coordinates property
    structure.coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Mock the residues property to ensure it passes the has_valid_residues check
    mock_residue = MagicMock()
    structure.residues = [mock_residue]

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


def test_get_coordinate_manager_inertia(mock_structure):
    """Test applying inertia-based transformation."""
    with (
        patch("flatprot.cli.commands.InertiaTransformer") as MockTransformer,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
    ):
        # Set up mocks
        mock_transformer = MockTransformer.return_value
        # Important: we need to properly set up the transform return value
        transformed_coords = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        mock_transformer.transform.return_value = transformed_coords

        mock_projector = MockProjector.return_value
        # Set up the projector return values
        canvas_coords = np.array([[20.0, 30.0], [50.0, 60.0]])
        depth_values = np.array([0.5, 0.7])
        mock_projector.project.return_value = (canvas_coords, depth_values)

        # Call the function with default parameters
        result = get_coordinate_manager(mock_structure)

        # Verify transformer was called correctly
        MockTransformer.assert_called_once()
        mock_transformer.transform.assert_called_once()

        # Check result structure
        assert isinstance(result, CoordinateManager)

        # Verify coordinates were added to the manager
        assert CoordinateType.COORDINATES in result.coordinates
        assert CoordinateType.TRANSFORMED in result.coordinates
        assert CoordinateType.CANVAS in result.coordinates
        assert CoordinateType.DEPTH in result.coordinates


def test_get_coordinate_manager_matrix(mock_structure, valid_matrix_file):
    """Test applying transformation with a custom matrix."""
    with (
        patch("flatprot.cli.commands.MatrixTransformer") as MockTransformer,
        patch("flatprot.cli.commands.MatrixLoader") as MockLoader,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
    ):
        # Set up mocks
        mock_transformer = MockTransformer.return_value
        mock_matrix = MagicMock(spec=TransformationMatrix)
        MockLoader.return_value.load.return_value = mock_matrix

        # Set up the transform return value
        transformed_coords = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        mock_transformer.transform.return_value = transformed_coords

        mock_projector = MockProjector.return_value
        # Set up the projector return values
        canvas_coords = np.array([[20.0, 30.0], [50.0, 60.0]])
        depth_values = np.array([0.5, 0.7])
        mock_projector.project.return_value = (canvas_coords, depth_values)

        # Call the function with matrix file
        result = get_coordinate_manager(mock_structure, valid_matrix_file)

        # Verify transformer was properly set up
        MockTransformer.assert_called_once()
        mock_transformer.transform.assert_called_once()

        # Check result structure
        assert isinstance(result, CoordinateManager)

        # Verify coordinates were added to the manager
        assert CoordinateType.COORDINATES in result.coordinates
        assert CoordinateType.TRANSFORMED in result.coordinates
        assert CoordinateType.CANVAS in result.coordinates
        assert CoordinateType.DEPTH in result.coordinates


def test_get_coordinate_manager_invalid_matrix(mock_structure, invalid_matrix_file):
    """Test handling of an invalid matrix file."""
    with (
        patch("flatprot.cli.commands.MatrixLoader") as MockLoader,
        patch("flatprot.cli.commands.console") as mock_console,
        patch("flatprot.cli.commands.InertiaTransformer") as MockInertiaTransformer,
        patch("flatprot.cli.commands.MatrixTransformer") as MockMatrixTransformer,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
    ):
        # Set up mocks
        MockLoader.return_value.load.side_effect = ValueError("Invalid matrix")

        # Set up the inertia transformer as fallback
        mock_inertia_transformer = MockInertiaTransformer.return_value
        transformed_coords = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        mock_inertia_transformer.transform.return_value = transformed_coords

        # We need to also set up the matrix transformer for the identity fallback
        mock_matrix_transformer = MockMatrixTransformer.return_value
        mock_matrix_transformer.transform.return_value = transformed_coords

        mock_projector = MockProjector.return_value
        canvas_coords = np.array([[20.0, 30.0], [50.0, 60.0]])
        depth_values = np.array([0.5, 0.7])
        mock_projector.project.return_value = (canvas_coords, depth_values)

        # Call the function with invalid matrix file
        result = get_coordinate_manager(mock_structure, invalid_matrix_file)

        # Should print a warning about matrix loading failure
        mock_console.print.assert_called()

        # Verify appropriate transformer was used as fallback
        # This depends on whether the has_valid_residues check passes in get_coordinate_manager
        assert MockInertiaTransformer.called or MockMatrixTransformer.called

        # Check result structure
        assert isinstance(result, CoordinateManager)

        # Verify coordinates were added to the manager
        assert CoordinateType.COORDINATES in result.coordinates
        assert CoordinateType.TRANSFORMED in result.coordinates
        assert CoordinateType.CANVAS in result.coordinates
        assert CoordinateType.DEPTH in result.coordinates


def test_get_coordinate_manager_no_coordinates():
    """Test handling of a structure with no coordinates."""
    with patch("flatprot.cli.commands.console") as mock_console:
        # Create a structure with no coordinates
        structure = MagicMock(spec=Structure)
        structure.coordinates = None

        # Call the function
        result = get_coordinate_manager(structure)

        # Should print a warning about skipping transformation
        mock_console.print.assert_called()

        # Verify result is an empty CoordinateManager
        assert isinstance(result, CoordinateManager)

        # Should have no coordinates added
        assert len(result.coordinates) == 0


def test_get_coordinate_manager_missing_file(mock_structure):
    """Test handling a missing matrix file."""
    with (
        patch("flatprot.cli.commands.console") as mock_console,
        patch("flatprot.cli.commands.InertiaTransformer") as MockInertiaTransformer,
        patch("flatprot.cli.commands.MatrixTransformer") as MockMatrixTransformer,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
    ):
        # Set up transformers
        mock_inertia_transformer = MockInertiaTransformer.return_value
        mock_matrix_transformer = MockMatrixTransformer.return_value
        transformed_coords = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        mock_inertia_transformer.transform.return_value = transformed_coords
        mock_matrix_transformer.transform.return_value = transformed_coords

        mock_projector = MockProjector.return_value
        canvas_coords = np.array([[20.0, 30.0], [50.0, 60.0]])
        depth_values = np.array([0.5, 0.7])
        mock_projector.project.return_value = (canvas_coords, depth_values)

        # Use a nonexistent path (this doesn't need to be mocked)
        nonexistent_path = Path("/tmp/definitely_nonexistent_file_for_testing.npy")

        # Call the function with a nonexistent matrix file
        result = get_coordinate_manager(mock_structure, nonexistent_path)

        # Should print a warning about missing file
        mock_console.print.assert_called()

        # Verify appropriate transformer was used as fallback
        # This depends on whether the has_valid_residues check passes in get_coordinate_manager
        assert MockInertiaTransformer.called or MockMatrixTransformer.called

        # Check result structure
        assert isinstance(result, CoordinateManager)

        # Verify coordinates were added to the manager
        assert CoordinateType.COORDINATES in result.coordinates
        assert CoordinateType.TRANSFORMED in result.coordinates
        assert CoordinateType.CANVAS in result.coordinates
        assert CoordinateType.DEPTH in result.coordinates


def test_main_with_default_transformation():
    """Test the main function with default transformation."""
    with (
        tempfile.NamedTemporaryFile(suffix=".pdb") as temp_pdb,
        tempfile.NamedTemporaryFile(suffix=".svg") as temp_svg,
        patch("flatprot.cli.commands.GemmiStructureParser") as MockParser,
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_manager,
        patch("flatprot.cli.commands.console"),
    ):
        # Create a dummy PDB file with minimal content
        with open(temp_pdb.name, "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
            )

        # Setup the mocks
        mock_structure = MagicMock(spec=Structure)
        mock_cm = MagicMock(spec=CoordinateManager)
        MockParser.return_value.parse_structure.return_value = mock_structure
        mock_manager.return_value = mock_cm  # Return mock CoordinateManager

        # Call the main function
        main(Path(temp_pdb.name), Path(temp_svg.name))

        # Verify function calls
        MockParser.return_value.parse_structure.assert_called_once_with(
            Path(temp_pdb.name)
        )
        mock_manager.assert_called_once_with(mock_structure, None)


def test_main_with_custom_transformation(valid_matrix_file):
    """Test the main function with custom transformation."""
    with (
        tempfile.NamedTemporaryFile(suffix=".pdb") as temp_pdb,
        tempfile.NamedTemporaryFile(suffix=".svg") as temp_svg,
        patch("flatprot.cli.commands.GemmiStructureParser") as MockParser,
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_manager,
        patch("flatprot.cli.commands.console"),
    ):
        # Create a dummy PDB file with minimal content
        with open(temp_pdb.name, "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
            )

        # Setup the mocks
        mock_structure = MagicMock(spec=Structure)
        mock_cm = MagicMock(spec=CoordinateManager)
        MockParser.return_value.parse_structure.return_value = mock_structure
        mock_manager.return_value = mock_cm  # Return mock CoordinateManager

        # Call the main function
        main(Path(temp_pdb.name), Path(temp_svg.name), matrix=valid_matrix_file)

        # Verify function calls
        MockParser.return_value.parse_structure.assert_called_once_with(
            Path(temp_pdb.name)
        )
        mock_manager.assert_called_once_with(mock_structure, valid_matrix_file)


def test_main_with_transformation_error():
    """Test the main function when transformation fails."""
    with (
        tempfile.NamedTemporaryFile(suffix=".pdb") as temp_pdb,
        tempfile.NamedTemporaryFile(suffix=".svg") as temp_svg,
        patch("flatprot.cli.commands.GemmiStructureParser") as MockParser,
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_manager,
        patch("flatprot.cli.commands.console"),
    ):
        # Create a dummy PDB file with minimal content
        with open(temp_pdb.name, "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
            )

        # Setup the mocks
        mock_structure = MagicMock(spec=Structure)
        MockParser.return_value.parse_structure.return_value = mock_structure
        mock_manager.side_effect = TransformationError("Test error")

        # Call the main function
        result = main(Path(temp_pdb.name), Path(temp_svg.name))

        # Verify that an error was raised and handled
        assert result == 1
