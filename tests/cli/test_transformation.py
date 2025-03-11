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
from flatprot.core.components import Structure
from flatprot.core import CoordinateManager


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
        patch("flatprot.cli.commands.CoordinateManager") as MockCoordinateManager,
        patch(
            "flatprot.cli.commands.OrthographicProjectionParameters"
        ) as MockProjectionParams,
    ):
        # Set up mocks
        mock_transformer = MockTransformer.return_value
        mock_transformer.transform.return_value = np.array(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
        )

        mock_projector = MockProjector.return_value
        mock_projector.project.return_value = (
            np.array([[20.0, 30.0], [50.0, 60.0]]),  # canvas coordinates
            np.array([0.5, 0.7]),  # depth values
        )

        mock_cm_instance = MockCoordinateManager.return_value

        # Call the function
        result = get_coordinate_manager(mock_structure)

        # Verify the result is a CoordinateManager
        assert result is mock_cm_instance

        # Check that the transformer was created and used correctly
        MockTransformer.assert_called_once()
        mock_transformer.transform.assert_called_once()

        # Check that the projector was created and used correctly
        MockProjector.assert_called_once()
        MockProjectionParams.assert_called_once()
        mock_projector.project.assert_called_once()

        # Check that coordinates were added to the manager
        assert (
            mock_cm_instance.add.call_count == 4
        )  # Original, transformed, canvas, depth


def test_get_coordinate_manager_matrix(mock_structure, valid_matrix_file):
    """Test applying transformation with a custom matrix."""
    with (
        patch("flatprot.cli.commands.MatrixTransformer") as MockTransformer,
        patch("flatprot.cli.commands.MatrixLoader") as MockLoader,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
        patch("flatprot.cli.commands.CoordinateManager") as MockCoordinateManager,
        patch(
            "flatprot.cli.commands.OrthographicProjectionParameters"
        ) as MockProjectionParams,
    ):
        # Set up mocks
        mock_transformer = MockTransformer.return_value
        mock_matrix = MagicMock(spec=TransformationMatrix)
        MockLoader.return_value.load.return_value = mock_matrix

        mock_transformer.transform.return_value = np.array(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
        )

        mock_projector = MockProjector.return_value
        mock_projector.project.return_value = (
            np.array([[20.0, 30.0], [50.0, 60.0]]),  # canvas coordinates
            np.array([0.5, 0.7]),  # depth values
        )

        mock_cm_instance = MockCoordinateManager.return_value

        # Call the function
        result = get_coordinate_manager(mock_structure, valid_matrix_file)

        # Verify the result is a CoordinateManager
        assert result is mock_cm_instance

        # Check that the transformer was created and used correctly
        MockTransformer.assert_called_once()

        # The transform call should include the coordinates and parameters
        mock_transformer.transform.assert_called_once()
        args, kwargs = mock_transformer.transform.call_args
        assert np.array_equal(args[0], mock_structure.coordinates.copy())

        # Check that the projector was created and used correctly
        MockProjector.assert_called_once()
        MockProjectionParams.assert_called_once()
        mock_projector.project.assert_called_once()

        # Check that coordinates were added to the manager
        assert (
            mock_cm_instance.add.call_count == 4
        )  # Original, transformed, canvas, depth


def test_get_coordinate_manager_invalid_matrix(mock_structure, invalid_matrix_file):
    """Test handling of an invalid matrix file."""
    with (
        patch("flatprot.cli.commands.MatrixLoader") as MockLoader,
        patch("flatprot.cli.commands.console") as mock_console,
        patch("flatprot.cli.commands.InertiaTransformer") as MockInertiaTransformer,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
        patch("flatprot.cli.commands.CoordinateManager") as MockCoordinateManager,
    ):
        # Set up mocks
        MockLoader.return_value.load.side_effect = ValueError("Invalid matrix")
        mock_transformer = MockInertiaTransformer.return_value
        mock_transformer.transform.return_value = np.array(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
        )

        mock_projector = MockProjector.return_value
        mock_projector.project.return_value = (
            np.array([[20.0, 30.0], [50.0, 60.0]]),  # canvas coordinates
            np.array([0.5, 0.7]),  # depth values
        )

        mock_cm_instance = MockCoordinateManager.return_value

        # Call the function - should fall back to inertia transformer
        result = get_coordinate_manager(mock_structure, invalid_matrix_file)

        # Should print a warning but continue with inertia transformer
        mock_console.print.assert_called()

        # Should return a coordinate manager
        assert result is mock_cm_instance

        # Check that the inertia transformer was used as fallback
        MockInertiaTransformer.assert_called_once()


def test_get_coordinate_manager_no_coordinates():
    """Test handling of a structure with no coordinates."""
    with (
        patch("flatprot.cli.commands.console") as mock_console,
        patch("flatprot.cli.commands.CoordinateManager") as MockCoordinateManager,
    ):
        # Create a structure with no coordinates
        structure = MagicMock(spec=Structure)
        structure.coordinates = None

        mock_cm_instance = MockCoordinateManager.return_value

        # Call the function
        result = get_coordinate_manager(structure)

        # Should print a warning and return an empty coordinate manager
        mock_console.print.assert_called()

        # Should return a coordinate manager
        assert result is mock_cm_instance

        # No transforms should be added
        assert not MockCoordinateManager.return_value.add.called


def test_get_coordinate_manager_missing_file(mock_structure):
    """Test handling a missing matrix file."""
    with (
        patch("flatprot.cli.commands.Path") as MockPath,
        patch("flatprot.cli.commands.console") as mock_console,
        patch("flatprot.cli.commands.InertiaTransformer") as MockInertiaTransformer,
        patch("flatprot.cli.commands.OrthographicProjector") as MockProjector,
        patch("flatprot.cli.commands.CoordinateManager") as MockCoordinateManager,
    ):
        # Set up mocks
        MockPath.return_value.exists.return_value = False

        mock_transformer = MockInertiaTransformer.return_value
        mock_transformer.transform.return_value = np.array(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
        )

        mock_projector = MockProjector.return_value
        mock_projector.project.return_value = (
            np.array([[20.0, 30.0], [50.0, 60.0]]),  # canvas coordinates
            np.array([0.5, 0.7]),  # depth values
        )

        mock_cm_instance = MockCoordinateManager.return_value

        # Call the function - should fall back to inertia transformer
        result = get_coordinate_manager(mock_structure, Path("nonexistent_file.npy"))

        # Should print a warning but continue with inertia transformer
        mock_console.print.assert_called()

        # Should return a coordinate manager
        assert result is mock_cm_instance

        # Check that the inertia transformer was used as fallback
        MockInertiaTransformer.assert_called_once()


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
