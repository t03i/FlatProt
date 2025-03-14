# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coordinate manager utility functions."""

from pathlib import Path

import numpy as np
import pytest

from flatprot.utils.coordinate_manger import (
    get_projection_parameters,
    apply_projection,
    create_transformer,
    create_coordinate_manager,
)
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.style import StyleManager
from flatprot.core.components import Structure
from flatprot.projection import OrthographicProjector, ProjectionError
from flatprot.transformation import (
    InertiaTransformer,
    MatrixTransformer,
    TransformationError,
    TransformationMatrix,
)


class TestGetProjectionParameters:
    """Tests for the get_projection_parameters function."""

    def test_default_parameters(self, mocker) -> None:
        """Test that default parameters are used when style manager lacks attributes."""
        # Mock style manager without canvas_style attribute
        style_manager = mocker.Mock(spec=StyleManager)

        # Call the function
        params = get_projection_parameters(style_manager)

        # Verify default values
        assert params.width == 800
        assert params.height == 600
        assert params.padding_x == 0.05
        assert params.padding_y == 0.05
        assert params.maintain_aspect_ratio is True

    def test_custom_parameters(self, mocker) -> None:
        """Test that custom parameters from style manager are used when available."""
        # Mock style manager with custom canvas style
        style_manager = mocker.Mock(spec=StyleManager)
        canvas_style = mocker.Mock()
        canvas_style.width = 1200
        canvas_style.height = 800
        canvas_style.padding_x = 0.1
        canvas_style.padding_y = 0.2
        canvas_style.maintain_aspect_ratio = False
        style_manager.canvas_style = canvas_style

        # Call the function
        params = get_projection_parameters(style_manager)

        # Verify custom values
        assert params.width == 1200
        assert params.height == 800
        assert params.padding_x == 0.1
        assert params.padding_y == 0.2
        assert params.maintain_aspect_ratio is False

    def test_partial_custom_parameters(self, mocker) -> None:
        """Test handling of partial canvas style parameters."""
        # Mock style manager with partial canvas style
        style_manager = mocker.Mock(spec=StyleManager)
        canvas_style = mocker.Mock()
        canvas_style.width = 1000  # Only set width
        canvas_style.height = None
        canvas_style.padding_x = None
        canvas_style.padding_y = None
        canvas_style.maintain_aspect_ratio = None
        style_manager.canvas_style = canvas_style

        # Call the function
        params = get_projection_parameters(style_manager)

        # Verify mixed values (custom width, default others)
        assert params.width == 1000
        assert params.height == 600  # default
        assert params.padding_x == 0.05  # default
        assert params.padding_y == 0.05  # default
        assert params.maintain_aspect_ratio is True  # default


class TestApplyProjection:
    """Tests for the apply_projection function."""

    def test_basic_projection(self, mocker) -> None:
        """Test basic projection of transformed coordinates."""
        # Mock dependencies
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)

        # Setup coordinate manager to have transformed coordinates
        coordinate_manager.has_type.return_value = True

        # Mock coordinate data structure
        transformed_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        coordinate_manager.coordinates = {
            CoordinateType.TRANSFORMED: {(0, 2): transformed_coords}
        }

        # Mock projector
        mock_projector = mocker.Mock(spec=OrthographicProjector)
        canvas_coords = np.array([[100, 200], [300, 400]])
        depth_values = np.array([3.0, 6.0])
        mock_projector.project.return_value = (canvas_coords, depth_values)

        # Mock projector and parameters
        mocker.patch(
            "flatprot.utils.coordinate_manger.OrthographicProjector",
            return_value=mock_projector,
        )
        mocker.patch(
            "flatprot.utils.coordinate_manger.get_projection_parameters",
            return_value=mocker.Mock(),
        )

        # Call function
        result = apply_projection(coordinate_manager, style_manager)

        # Verify projector was called correctly
        mock_projector.project.assert_called_once()

        # Verify coordinates were added properly - capture the calls instead of using assert_any_call
        assert (
            coordinate_manager.add.call_count == 2
        ), "Should call add twice for canvas and depth"

        # Get the two calls
        call_args_list = coordinate_manager.add.call_args_list

        # Check first call (canvas coordinates)
        canvas_call = call_args_list[0]
        assert canvas_call[0][0] == 0, "Start index should be 0"
        assert canvas_call[0][1] == 2, "End index should be 2"
        assert np.array_equal(
            canvas_call[0][2], canvas_coords
        ), "Canvas coordinates should match"
        assert (
            canvas_call[0][3] == CoordinateType.CANVAS
        ), "Coordinate type should be CANVAS"

        # Check second call (depth values)
        depth_call = call_args_list[1]
        assert depth_call[0][0] == 0, "Start index should be 0"
        assert depth_call[0][1] == 2, "End index should be 2"
        assert np.array_equal(
            depth_call[0][2], depth_values
        ), "Depth values should match"
        assert (
            depth_call[0][3] == CoordinateType.DEPTH
        ), "Coordinate type should be DEPTH"

        # Verify result
        assert result == coordinate_manager

    def test_no_transformed_coordinates(self, mocker) -> None:
        """Test handling when no transformed coordinates are available."""
        # Mock dependencies
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)

        # Setup coordinate manager to have no transformed coordinates
        coordinate_manager.has_type.return_value = False

        # Mock console output
        mock_console = mocker.patch("flatprot.utils.coordinate_manger.console.print")

        # Call function
        result = apply_projection(coordinate_manager, style_manager)

        # Verify warning was printed
        mock_console.assert_called_once()
        assert "Warning" in mock_console.call_args[0][0]
        assert "No transformed coordinates" in mock_console.call_args[0][0]

        # Verify result
        assert result == coordinate_manager

    def test_projection_error_handling(self, mocker) -> None:
        """Test error handling during projection."""
        # Mock dependencies
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)

        # Setup coordinate manager to have transformed coordinates
        coordinate_manager.has_type.return_value = True

        # Mock coordinate data structure
        transformed_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        coordinate_manager.coordinates = {
            CoordinateType.TRANSFORMED: {(0, 2): transformed_coords}
        }

        # Mock projector to raise error
        mock_projector = mocker.Mock(spec=OrthographicProjector)
        mock_projector.project.side_effect = Exception("Test projection error")

        # Mock projector and parameters
        mocker.patch(
            "flatprot.utils.coordinate_manger.OrthographicProjector",
            return_value=mock_projector,
        )
        mocker.patch(
            "flatprot.utils.coordinate_manger.get_projection_parameters",
            return_value=mocker.Mock(),
        )

        # Mock console output
        mocker.patch("flatprot.utils.coordinate_manger.console.print")

        # Call function and expect exception
        with pytest.raises(ProjectionError) as exc_info:
            apply_projection(coordinate_manager, style_manager)

        # Verify error message
        assert "Failed to apply projection" in str(exc_info.value)
        assert "Test projection error" in str(exc_info.value)


class TestCreateTransformer:
    """Tests for the create_transformer function."""

    def test_with_matrix_path(self, mocker) -> None:
        """Test creating transformer using a matrix path."""
        # Mock structure with properly configured residues
        structure = mocker.Mock(spec=Structure)

        # Configure the residues attribute to respond to len()
        mock_residues = mocker.Mock()
        mock_residues.__len__ = mocker.Mock(return_value=2)  # Make it respond to len()
        structure.residues = mock_residues

        # Mock matrix loading
        mock_matrix = mocker.Mock(spec=TransformationMatrix)
        mock_matrix_loader = mocker.Mock()
        mock_matrix_loader.load.return_value = mock_matrix

        mocker.patch(
            "flatprot.utils.coordinate_manger.MatrixLoader",
            return_value=mock_matrix_loader,
        )

        # Call function with matrix path
        matrix_path = Path("/path/to/matrix.npy")
        transformer, params = create_transformer(structure, matrix_path)

        # Verify results
        assert isinstance(transformer, MatrixTransformer)
        assert params.matrix == mock_matrix

    def test_with_inertia_transformer(self, mocker) -> None:
        """Test falling back to inertia transformer when no matrix is provided."""
        # Mock structure with coordinates and properly configured residues
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Configure residues with mock list that responds to len()
        mock_residues = ["RES1", "RES2"]  # This will work with len()
        structure.residues = mock_residues

        # Call function without matrix path
        transformer, params = create_transformer(structure)

        # Verify results
        assert isinstance(transformer, InertiaTransformer)
        assert params.residues == structure.residues

    def test_fallback_to_identity_matrix(self, mocker) -> None:
        """Test falling back to identity matrix when structure lacks required properties."""
        # Mock structure without coordinates
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = None

        # Configure residues as empty list (responds to len() and returns 0)
        structure.residues = []

        # Mock console output
        mocker.patch("flatprot.utils.coordinate_manger.console.print")

        # Call function
        transformer, params = create_transformer(structure)

        # Verify results
        assert isinstance(transformer, MatrixTransformer)
        assert isinstance(params.matrix, TransformationMatrix)
        assert np.array_equal(params.matrix.rotation, np.eye(3))
        assert np.array_equal(params.matrix.translation, np.zeros(3))

    def test_matrix_loading_error(self, mocker) -> None:
        """Test handling of errors when loading transformation matrix."""
        # Mock structure with coordinates and properly configured residues
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        structure.residues = ["RES1", "RES2"]  # This works with len()

        # Mock matrix loader to raise exception
        mocker.patch(
            "flatprot.utils.coordinate_manger.MatrixLoader",
            side_effect=Exception("Test matrix loading error"),
        )

        # Mock console output
        mocker.patch("flatprot.utils.coordinate_manger.console.print")

        # Call function with matrix path
        matrix_path = Path("/path/to/matrix.npy")
        transformer, params = create_transformer(structure, matrix_path)

        # Verify fallback to inertia transformer
        assert isinstance(transformer, InertiaTransformer)
        assert params.residues == structure.residues


class TestCreateCoordinateManager:
    """Tests for the create_coordinate_manager function."""

    def test_basic_coordinate_manager(self, mocker) -> None:
        """Test basic creation of coordinate manager with transformation."""
        # Mock structure with coordinates
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Mock transformer and transformed coordinates
        mock_transformer = mocker.Mock()
        mock_params = mocker.Mock()
        transformed_coords = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        mock_transformer.transform.return_value = transformed_coords

        # Mock create_transformer to return our mocks
        mocker.patch(
            "flatprot.utils.coordinate_manger.create_transformer",
            return_value=(mock_transformer, mock_params),
        )

        # Call function
        result = create_coordinate_manager(structure)

        # Verify transformer was called
        mock_transformer.transform.assert_called_once()

        # Check if original coordinates were added
        assert result.has_type(CoordinateType.COORDINATES)
        coords = result.get(0, 2, CoordinateType.COORDINATES)
        assert np.array_equal(coords, structure.coordinates)

        # Check if transformed coordinates were added
        assert result.has_type(CoordinateType.TRANSFORMED)
        transformed = result.get(0, 2, CoordinateType.TRANSFORMED)
        assert np.array_equal(transformed, transformed_coords)

    def test_no_coordinates(self, mocker) -> None:
        """Test handling when structure has no coordinates."""
        # Mock structure without coordinates
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = None

        # Mock console output
        mocker.patch("flatprot.utils.coordinate_manger.console.print")

        # Call function
        result = create_coordinate_manager(structure)

        # Verify result is empty coordinate manager
        assert isinstance(result, CoordinateManager)
        assert not result.has_type(CoordinateType.COORDINATES)
        assert not result.has_type(CoordinateType.TRANSFORMED)

    def test_transformation_error(self, mocker) -> None:
        """Test error handling during transformation."""
        # Mock structure with coordinates
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Mock transformer to raise exception
        mock_transformer = mocker.Mock()
        mock_params = mocker.Mock()
        mock_transformer.transform.side_effect = Exception("Test transformation error")

        # Mock create_transformer to return our mocks
        mocker.patch(
            "flatprot.utils.coordinate_manger.create_transformer",
            return_value=(mock_transformer, mock_params),
        )

        # Mock console output
        mocker.patch("flatprot.utils.coordinate_manger.console.print")

        # Call function and expect exception
        with pytest.raises(TransformationError) as exc_info:
            create_coordinate_manager(structure)

        # Verify error message
        assert "Failed to apply transformation" in str(exc_info.value)
        assert "Test transformation error" in str(exc_info.value)
