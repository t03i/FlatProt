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
from flatprot.core import (
    CoordinateManager,
    CoordinateType,
    ResidueRange,
)
from flatprot.style import StyleManager, StyleType, CanvasStyle
from flatprot.core.components import (
    Structure,
    Chain,
    ResidueType,
)  # Import Chain and Residue for mocking
from flatprot.projection import (
    OrthographicProjector,
    ProjectionError,
    OrthographicProjectionParameters,  # Import specific parameters class
)
from flatprot.transformation import (
    InertiaTransformer,
    MatrixTransformer,
    TransformationError,
    TransformationMatrix,
    BaseTransformer,  # Import BaseTransformer
    MatrixTransformParameters,  # Import specific parameters class
    InertiaTransformParameters,  # Import specific parameters class
)

# Import MatrixLoader for mocking
from flatprot.io import MatrixLoader


class TestGetProjectionParameters:
    """Tests for the get_projection_parameters function."""

    def test_default_parameters(self, mocker) -> None:
        """Test that default parameters are returned when using default style."""
        style_manager = mocker.Mock(spec=StyleManager)
        # Mock the return value of get_style for CANVAS type
        default_canvas_style = CanvasStyle()  # Create default style instance
        style_manager.get_style.return_value = default_canvas_style

        params = get_projection_parameters(style_manager)

        # Verify get_style was called correctly
        style_manager.get_style.assert_called_once_with(StyleType.CANVAS)
        # Verify parameters match the default canvas style
        expected_params = OrthographicProjectionParameters(
            width=default_canvas_style.width,
            height=default_canvas_style.height,
            padding_x=default_canvas_style.padding_x,
            padding_y=default_canvas_style.padding_y,
            maintain_aspect_ratio=default_canvas_style.maintain_aspect_ratio,
        )
        # Can't directly compare objects with numpy arrays inside
        assert params.width == expected_params.width
        assert params.height == expected_params.height
        assert params.padding_x == expected_params.padding_x
        assert params.padding_y == expected_params.padding_y
        assert params.maintain_aspect_ratio == expected_params.maintain_aspect_ratio
        assert np.array_equal(params.view_direction, expected_params.view_direction)
        assert np.array_equal(params.up_vector, expected_params.up_vector)

    def test_custom_parameters(self, mocker) -> None:
        """Test that custom parameters are used when get_style returns custom style."""
        style_manager = mocker.Mock(spec=StyleManager)
        # Mock the return value of get_style for CANVAS type with custom values
        custom_canvas_style = mocker.Mock(spec=CanvasStyle)
        custom_canvas_style.width = 1200
        custom_canvas_style.height = 800
        custom_canvas_style.padding_x = 0.1
        custom_canvas_style.padding_y = 0.2
        custom_canvas_style.maintain_aspect_ratio = False
        style_manager.get_style.return_value = custom_canvas_style

        params = get_projection_parameters(style_manager)

        # Verify get_style was called correctly
        style_manager.get_style.assert_called_once_with(StyleType.CANVAS)
        # Verify parameters match the custom canvas style
        assert params.width == 1200
        assert params.height == 800
        assert params.padding_x == 0.1
        assert params.padding_y == 0.2
        assert params.maintain_aspect_ratio is False


class TestApplyProjection:
    """Tests for the apply_projection function."""

    def test_basic_projection(self, mocker) -> None:
        """Test basic projection of transformed coordinates."""
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)

        coordinate_manager.has_type.return_value = True
        transformed_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Update coordinate dict structure: Key is a tuple (start, end)
        coordinate_manager.coordinates = {
            CoordinateType.TRANSFORMED: {(0, 2): transformed_coords}
        }

        mock_projector = mocker.Mock(spec=OrthographicProjector)
        canvas_coords = np.array([[100, 200], [300, 400]])
        depth_values = np.array([3.0, 6.0])  # Depth from z-coordinates
        mock_projector.project.return_value = (canvas_coords, depth_values)

        # Mock projector class and parameter function
        mock_ortho_projector_cls = mocker.patch(
            "flatprot.utils.coordinate_manger.OrthographicProjector",
            return_value=mock_projector,
        )
        mock_projection_params = mocker.Mock(spec=OrthographicProjectionParameters)
        mock_get_params = mocker.patch(
            "flatprot.utils.coordinate_manger.get_projection_parameters",
            return_value=mock_projection_params,
        )

        result = apply_projection(coordinate_manager, style_manager)

        # Verify mocks
        mock_ortho_projector_cls.assert_called_once()
        mock_get_params.assert_called_once_with(style_manager)
        # Verify project called with the specific range's coordinates
        mock_projector.project.assert_called_once()
        np.testing.assert_array_equal(
            mock_projector.project.call_args[0][0], transformed_coords
        )
        assert mock_projector.project.call_args[0][1] is mock_projection_params

        # Verify coordinates were added with the correct range (0, 2)
        expected_calls = [
            mocker.call(0, 2, mocker.ANY, CoordinateType.CANVAS),
            mocker.call(0, 2, mocker.ANY, CoordinateType.DEPTH),
        ]
        coordinate_manager.add.assert_has_calls(expected_calls, any_order=True)
        # Explicitly check array contents in the calls
        canvas_call_args = coordinate_manager.add.call_args_list[0][0]
        depth_call_args = coordinate_manager.add.call_args_list[1][0]
        # Find which call corresponds to CANVAS vs DEPTH (order might vary)
        if canvas_call_args[3] == CoordinateType.CANVAS:
            np.testing.assert_array_equal(canvas_call_args[2], canvas_coords)
            np.testing.assert_array_equal(depth_call_args[2], depth_values)
        else:
            np.testing.assert_array_equal(canvas_call_args[2], depth_values)
            np.testing.assert_array_equal(depth_call_args[2], canvas_coords)

        assert result is coordinate_manager

    def test_no_transformed_coordinates(self, mocker) -> None:
        """Test handling when no transformed coordinates are available."""
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)
        coordinate_manager.has_type.return_value = False
        # Ensure coordinates dict doesn't contain TRANSFORMED type
        coordinate_manager.coordinates = {CoordinateType.COORDINATES: {}}

        mock_logger_warning = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.warning"
        )
        # Mock projector/params functions to ensure they aren't called
        mock_ortho_projector_cls = mocker.patch(
            "flatprot.utils.coordinate_manger.OrthographicProjector"
        )
        mock_get_params = mocker.patch(
            "flatprot.utils.coordinate_manger.get_projection_parameters"
        )

        result = apply_projection(coordinate_manager, style_manager)

        mock_logger_warning.assert_called_once_with(
            "No transformed coordinates to project"
        )
        mock_ortho_projector_cls.assert_not_called()
        mock_get_params.assert_not_called()
        coordinate_manager.add.assert_not_called()
        assert result is coordinate_manager

    def test_projection_error_handling(self, mocker) -> None:
        """Test error handling during projection."""
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)
        coordinate_manager.has_type.return_value = True
        transformed_coords = np.array([[1.0, 2.0, 3.0]])
        coordinate_manager.coordinates = {
            CoordinateType.TRANSFORMED: {(0, 1): transformed_coords}
        }

        mock_projector = mocker.Mock(spec=OrthographicProjector)
        projection_exception = Exception("Test projection error")
        mock_projector.project.side_effect = projection_exception

        mocker.patch(
            "flatprot.utils.coordinate_manger.OrthographicProjector",
            return_value=mock_projector,
        )
        mocker.patch(
            "flatprot.utils.coordinate_manger.get_projection_parameters",
            return_value=mocker.Mock(),
        )
        mock_logger_error = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.error"
        )

        with pytest.raises(ProjectionError) as exc_info:
            apply_projection(coordinate_manager, style_manager)

        # Check that the raised error includes the original message
        assert str(projection_exception) in str(exc_info.value)
        # Check that the error was logged correctly
        mock_logger_error.assert_called_once()
        assert str(projection_exception) in mock_logger_error.call_args[0][0]


class TestCreateTransformer:
    """Tests for the create_transformer function."""

    def test_with_matrix_path(self, mocker) -> None:
        """Test creating transformer using a matrix path."""
        # Structure mock needed minimally for potential fallback logging
        structure = mocker.Mock(spec=Structure)
        structure.residues = [mocker.Mock()]

        # Mock MatrixLoader
        mock_matrix = mocker.Mock(spec=TransformationMatrix)
        mock_matrix_loader = mocker.Mock(spec=MatrixLoader)
        mock_matrix_loader.load.return_value = mock_matrix
        mock_matrix_loader_cls = mocker.patch(
            "flatprot.utils.coordinate_manger.MatrixLoader",
            return_value=mock_matrix_loader,
        )

        matrix_path = Path("/path/to/matrix.npy")
        transformer, params = create_transformer(structure, matrix_path)

        mock_matrix_loader_cls.assert_called_once_with(matrix_path)
        mock_matrix_loader.load.assert_called_once()
        # Verify correct transformer and parameters class
        assert isinstance(transformer, MatrixTransformer)
        assert isinstance(params, MatrixTransformParameters)
        assert params.matrix is mock_matrix

    def test_with_inertia_transformer(self, mocker) -> None:
        """Test creating inertia transformer when structure is valid."""
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array([[1.0, 2.0, 3.0]])
        # Mock residues list to satisfy len > 0 check
        structure.residues = [mocker.Mock()]

        transformer, params = create_transformer(structure)

        # Verify correct transformer and parameters class
        assert isinstance(transformer, InertiaTransformer)
        assert isinstance(params, InertiaTransformParameters)
        assert params.residues is structure.residues

    def test_fallback_to_identity_matrix(self, mocker) -> None:
        """Test falling back to identity matrix when structure lacks properties."""
        # Case 1: No coordinates
        structure_no_coords = mocker.Mock(
            spec=Structure, coordinates=None, residues=[mocker.Mock()]
        )
        # Case 2: No residues (empty list)
        structure_no_residues = mocker.Mock(
            spec=Structure, coordinates=np.array([[1.0]]), residues=[]
        )
        # Case 3: Residues is None
        structure_none_residues = mocker.Mock(
            spec=Structure, coordinates=np.array([[1.0]]), residues=None
        )

        mock_logger_warning = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.warning"
        )

        for struct in [
            structure_no_coords,
            structure_no_residues,
            structure_none_residues,
        ]:
            transformer, params = create_transformer(struct)
            assert isinstance(transformer, MatrixTransformer)
            assert isinstance(params, MatrixTransformParameters)
            np.testing.assert_array_equal(params.matrix.rotation, np.eye(3))
            np.testing.assert_array_equal(params.matrix.translation, np.zeros(3))

        # Check warning was logged for each invalid case
        assert mock_logger_warning.call_count == 3
        mock_logger_warning.assert_called_with(
            "Structure lacks required properties for inertia transformation, using identity matrix"
        )

    def test_matrix_loading_error_fallback_inertia(self, mocker) -> None:
        """Test fallback to inertia when matrix loading fails and structure is valid."""
        # Valid structure for inertia fallback
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array([[1.0, 2.0, 3.0]])
        structure.residues = [mocker.Mock()]

        # Mock MatrixLoader to raise error
        load_exception = Exception("Load Fail")
        mock_matrix_loader = mocker.Mock(spec=MatrixLoader)
        mock_matrix_loader.load.side_effect = load_exception
        mocker.patch(
            "flatprot.utils.coordinate_manger.MatrixLoader",
            return_value=mock_matrix_loader,
        )
        mock_logger_warning = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.warning"
        )

        matrix_path = Path("/fail/matrix.npy")
        transformer, params = create_transformer(structure, matrix_path)

        # Verify warnings and fallback type
        assert mock_logger_warning.call_count == 2
        mock_logger_warning.assert_any_call(
            f"Failed to load matrix from {matrix_path}: {load_exception}"
        )
        mock_logger_warning.assert_any_call(
            "Falling back to inertia-based transformation"
        )
        assert isinstance(transformer, InertiaTransformer)
        assert isinstance(params, InertiaTransformParameters)
        assert params.residues is structure.residues

    def test_matrix_loading_error_fallback_identity(self, mocker) -> None:
        """Test fallback to identity when matrix loading fails and structure is invalid."""
        # Invalid structure for inertia fallback (no coordinates)
        structure = mocker.Mock(
            spec=Structure, coordinates=None, residues=[mocker.Mock()]
        )

        # Mock MatrixLoader to raise error
        load_exception = Exception("Load Fail")
        mock_matrix_loader = mocker.Mock(spec=MatrixLoader)
        mock_matrix_loader.load.side_effect = load_exception
        mocker.patch(
            "flatprot.utils.coordinate_manger.MatrixLoader",
            return_value=mock_matrix_loader,
        )
        mock_logger_warning = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.warning"
        )

        matrix_path = Path("/fail/matrix.npy")
        transformer, params = create_transformer(structure, matrix_path)

        # Verify warnings (load fail, inertia attempt, identity fallback)
        assert mock_logger_warning.call_count == 3
        mock_logger_warning.assert_any_call(
            f"Failed to load matrix from {matrix_path}: {load_exception}"
        )
        mock_logger_warning.assert_any_call(
            "Falling back to inertia-based transformation"
        )
        mock_logger_warning.assert_any_call(
            "Structure lacks required properties for inertia transformation, using identity matrix"
        )
        # Verify fallback to identity matrix
        assert isinstance(transformer, MatrixTransformer)
        assert isinstance(params, MatrixTransformParameters)
        np.testing.assert_array_equal(params.matrix.rotation, np.eye(3))
        np.testing.assert_array_equal(params.matrix.translation, np.zeros(3))


class TestCreateCoordinateManager:
    """Tests for the create_coordinate_manager function."""

    @pytest.fixture
    def mock_structure(self, mocker) -> Structure:
        """Creates a mock Structure object with chains and residues."""
        structure = mocker.Mock(spec=Structure)
        structure.coordinates = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],  # Chain A
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],  # Chain B
            ]
        )

        # Mock Residues with numbers
        res_a1 = mocker.Mock(spec=ResidueType)
        res_a1.number = 10
        res_a2 = mocker.Mock(spec=ResidueType)
        res_a2.number = 11
        res_b1 = mocker.Mock(spec=ResidueType)
        res_b1.number = 25
        res_b2 = mocker.Mock(spec=ResidueType)
        res_b2.number = 26

        # Mock Chains with IDs and residues
        chain_a = mocker.Mock(spec=Chain)
        chain_a.id = "A"
        chain_a.residues = [res_a1, res_a2]
        chain_b = mocker.Mock(spec=Chain)
        chain_b.id = "B"
        chain_b.residues = [res_b1, res_b2]

        structure.__iter__ = lambda self: iter([chain_a, chain_b])
        return structure

    @pytest.fixture
    def mock_coord_manager(self, mocker) -> CoordinateManager:
        """Mocks the CoordinateManager class."""
        mock_instance = mocker.Mock(spec=CoordinateManager)
        mocker.patch(
            "flatprot.utils.coordinate_manger.CoordinateManager",
            return_value=mock_instance,
        )
        return mock_instance  # Return the mocked instance

    def test_basic_coordinate_manager(
        self, mocker, mock_structure, mock_coord_manager
    ) -> None:
        """Test basic creation with transformation using InertiaTransformer."""
        # Mock transformer and transformed coordinates
        mock_transformer = mocker.Mock(spec=BaseTransformer)  # Use BaseTransformer
        mock_params = mocker.Mock(
            spec=InertiaTransformParameters
        )  # Assume Inertia for default
        transformed_coords = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0],
                [100.0, 110.0, 120.0],
            ]
        )
        mock_transformer.transform.return_value = transformed_coords

        # Mock create_transformer to return Inertia transformer by default
        # Assign the result of the patch to a variable to use for assertions
        create_transformer_mock = mocker.patch(
            "flatprot.utils.coordinate_manger.create_transformer",
            return_value=(mock_transformer, mock_params),
        )
        # Mock ResidueRangeSet class to check its instantiation
        mock_range_set_cls = mocker.patch(
            "flatprot.utils.coordinate_manger.ResidueRangeSet"
        )
        mock_range_set_instance = (
            mock_range_set_cls.return_value
        )  # Capture the instance

        # Call function (matrix_path=None uses create_transformer default)
        result = create_coordinate_manager(mock_structure)

        # Verify create_transformer was called
        create_transformer_mock.assert_called_once_with(mock_structure, None)

        # Verify ResidueRangeSet was created correctly
        expected_ranges = [
            ResidueRange(chain_id="A", start=10, end=11),
            ResidueRange(chain_id="B", start=25, end=26),
        ]
        # Check the arguments passed to ResidueRangeSet constructor
        actual_ranges_arg = mock_range_set_cls.call_args[0][0]
        assert len(actual_ranges_arg) == len(expected_ranges)
        for actual, expected in zip(actual_ranges_arg, expected_ranges):
            assert actual.chain_id == expected.chain_id
            assert actual.start == expected.start
            assert actual.end == expected.end

        # Verify transformer.transform was called
        mock_transformer.transform.assert_called_once()
        # Check transform args (need to compare numpy arrays correctly)
        assert np.array_equal(
            mock_transformer.transform.call_args[0][0], mock_structure.coordinates
        )
        assert mock_transformer.transform.call_args[0][1] is mock_params

        # Verify add_range_set was called twice with the correct RangeSet instance
        assert mock_coord_manager.add_range_set.call_count == 2
        expected_add_calls = [
            mocker.call(
                mock_range_set_instance,
                mock_structure.coordinates,
                CoordinateType.COORDINATES,
            ),
            mocker.call(
                mock_range_set_instance, transformed_coords, CoordinateType.TRANSFORMED
            ),
        ]
        # Use assert_has_calls for checking multiple calls
        mock_coord_manager.add_range_set.assert_has_calls(
            expected_add_calls, any_order=False
        )  # Order matters here

        # Verify result is the mocked coordinate manager instance
        assert result is mock_coord_manager

    def test_no_coordinates(self, mocker, mock_coord_manager) -> None:
        """Test handling when structure has no coordinates."""
        # Mock structure without coordinates
        structure_no_coords = mocker.Mock(spec=Structure)
        structure_no_coords.coordinates = None
        structure_no_coords.__iter__ = lambda self: iter([])

        mock_logger_warning = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.warning"
        )
        # Ensure create_transformer is NOT called
        mock_create_transformer = mocker.patch(
            "flatprot.utils.coordinate_manger.create_transformer"
        )

        # Call function
        result = create_coordinate_manager(structure_no_coords)

        # Verify warning and that no transformation/adding happened
        mock_logger_warning.assert_called_once_with(
            "Structure has no coordinates, skipping transformation"
        )
        mock_create_transformer.assert_not_called()
        mock_coord_manager.add_range_set.assert_not_called()

        # Verify result is the mocked (empty) coordinate manager instance
        assert result is mock_coord_manager

    def test_transformation_error(
        self, mocker, mock_structure, mock_coord_manager
    ) -> None:
        """Test error handling during transformation step."""
        # Mock transformer to raise exception during transform
        mock_transformer = mocker.Mock(spec=BaseTransformer)
        mock_params = mocker.Mock()  # Type doesn't strictly matter here
        transform_exception = Exception("Test transformation error")
        mock_transformer.transform.side_effect = transform_exception

        # Mock create_transformer to return the error-raising transformer
        # Assign the result of the patch to a variable to use for assertions
        mock_create_transformer = mocker.patch(
            "flatprot.utils.coordinate_manger.create_transformer",
            return_value=(mock_transformer, mock_params),
        )
        # Mock logger
        mock_logger_error = mocker.patch(
            "flatprot.utils.coordinate_manger.logger.error"
        )
        # Mock ResidueRangeSet instantiation (happens before transform)
        mock_range_set_cls = mocker.patch(
            "flatprot.utils.coordinate_manger.ResidueRangeSet"
        )
        mock_range_set_instance = mock_range_set_cls.return_value

        # Call function and expect TransformationError
        with pytest.raises(TransformationError) as exc_info:
            create_coordinate_manager(mock_structure)

        # Verify error message includes the original error
        assert "Failed to apply transformation: Test transformation error" in str(
            exc_info.value
        )

        # Verify create_transformer was called
        mock_create_transformer.assert_called_once_with(mock_structure, None)
        # Verify ResidueRangeSet was created
        mock_range_set_cls.assert_called_once()
        # Verify original coordinates were added
        mock_coord_manager.add_range_set.assert_called_once_with(
            mock_range_set_instance,
            mock_structure.coordinates,
            CoordinateType.COORDINATES,
        )
        # Verify transform was called
        mock_transformer.transform.assert_called_once()
        # Verify error was logged
        mock_logger_error.assert_called_once()
        assert (
            "Error applying transformation: Test transformation error"
            in mock_logger_error.call_args[0][0]
        )
