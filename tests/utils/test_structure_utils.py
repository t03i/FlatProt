# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structure utility functions."""

import pytest
from pytest_mock import MockerFixture
import numpy as np
from pathlib import Path
from typing import Callable

# Core components
from flatprot.core import Structure

# IO components and errors
from flatprot.io import (
    MatrixLoader,
)
from flatprot.io.errors import (
    MatrixFileNotFoundError,
    MatrixFileError,
)  # Correct import path

# Projection components and errors
from flatprot.projection import (
    OrthographicProjection,
    OrthographicProjectionParameters,
)
from flatprot.projection import ProjectionError  # Correct import path

# Transformation components and errors
from flatprot.transformation import (
    InertiaTransformer,
    InertiaTransformationParameters,
    InertiaTransformationArguments,
    MatrixTransformer,
    MatrixTransformParameters,
    TransformationMatrix,
)
from flatprot.transformation.error import TransformationError  # Correct import path

# Functions to test
from flatprot.utils.structure_utils import (
    _load_transformation_matrix,  # Can test directly or indirectly
    transform_structure_with_matrix,
    transform_structure_with_inertia,
    project_structure_orthographically,
)

# --- Fixtures ---


@pytest.fixture
def mock_structure_coords(mocker: MockerFixture) -> Structure:
    """Provides a mock Structure with mock coordinates and apply_vectorized_transformation."""
    mock_struct = mocker.MagicMock(spec=Structure)
    mock_struct.id = "mock_struct_coords"
    # Simulate having some coordinates
    mock_struct.coordinates = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    # Mock the transformation application method
    mock_struct.apply_vectorized_transformation = mocker.MagicMock()
    # Simulate having residues for inertia tests
    mock_struct.residues = [
        mocker.MagicMock(),
        mocker.MagicMock(),
    ]  # Assume ResidueType mocks
    return mock_struct


@pytest.fixture
def identity_matrix() -> TransformationMatrix:
    """Provides an identity TransformationMatrix."""
    return TransformationMatrix(rotation=np.eye(3), translation=np.zeros(3))


@pytest.fixture
def non_identity_matrix() -> TransformationMatrix:
    """Provides a non-identity TransformationMatrix."""
    # Example: rotate 90 degrees around Z and translate
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([10.0, 20.0, 30.0])
    return TransformationMatrix(rotation=rotation, translation=translation)


@pytest.fixture
def mock_transformed_structure(mocker: MockerFixture) -> Structure:
    """Provides a mock Structure representing a transformed result."""
    mock_struct = mocker.MagicMock(spec=Structure)
    mock_struct.id = "transformed_struct"
    mock_struct.coordinates = np.array(
        [[11.0, 18.0, 33.0], [14.0, 15.0, 36.0]]
    )  # Dummy transformed coords
    return mock_struct


# --- Test Functions ---

# Tests for _load_transformation_matrix


def test_load_transformation_matrix_success(
    mocker: MockerFixture, non_identity_matrix: TransformationMatrix
) -> None:
    """Test loading a valid transformation matrix."""
    mock_loader_instance = mocker.MagicMock(spec=MatrixLoader)
    mock_loader_instance.load.return_value = non_identity_matrix
    mock_loader_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixLoader", return_value=mock_loader_instance
    )
    dummy_path = Path("dummy/matrix.npy")

    result_matrix = _load_transformation_matrix(dummy_path)

    mock_loader_cls.assert_called_once_with(dummy_path)
    mock_loader_instance.load.assert_called_once()
    assert result_matrix == non_identity_matrix


def test_load_transformation_matrix_file_not_found(
    mocker: MockerFixture, identity_matrix: TransformationMatrix
) -> None:
    """Test fallback to identity matrix when file is not found."""
    mock_loader_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixLoader",
        side_effect=MatrixFileNotFoundError("dummy/matrix.npy"),
    )
    dummy_path = Path("dummy/matrix.npy")

    result_matrix = _load_transformation_matrix(dummy_path)

    mock_loader_cls.assert_called_once_with(dummy_path)
    # Check if the returned matrix is equivalent to the identity matrix
    assert np.array_equal(result_matrix.rotation, identity_matrix.rotation)
    assert np.array_equal(result_matrix.translation, identity_matrix.translation)


def test_load_transformation_matrix_load_error(
    mocker: MockerFixture, identity_matrix: TransformationMatrix
) -> None:
    """Test fallback to identity matrix on MatrixFileError."""
    mock_loader_instance = mocker.MagicMock(spec=MatrixLoader)
    mock_loader_instance.load.side_effect = MatrixFileError("Load failed")
    mock_loader_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixLoader", return_value=mock_loader_instance
    )
    dummy_path = Path("dummy/matrix.npy")

    result_matrix = _load_transformation_matrix(dummy_path)

    mock_loader_cls.assert_called_once_with(dummy_path)
    mock_loader_instance.load.assert_called_once()
    # Check if the returned matrix is equivalent to the identity matrix
    assert np.array_equal(result_matrix.rotation, identity_matrix.rotation)
    assert np.array_equal(result_matrix.translation, identity_matrix.translation)


def test_load_transformation_matrix_unexpected_error(mocker: MockerFixture) -> None:
    """Test re-raising TransformationError for unexpected load issues."""
    mock_loader_instance = mocker.MagicMock(spec=MatrixLoader)
    mock_loader_instance.load.side_effect = Exception("Unexpected disk error")
    mock_loader_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixLoader", return_value=mock_loader_instance
    )
    dummy_path = Path("dummy/matrix.npy")

    with pytest.raises(TransformationError, match="Failed to load matrix file"):
        _load_transformation_matrix(dummy_path)

    mock_loader_cls.assert_called_once_with(dummy_path)
    mock_loader_instance.load.assert_called_once()


def test_load_transformation_matrix_no_path(
    identity_matrix: TransformationMatrix,
) -> None:
    """Test using identity matrix when no path is provided."""
    result_matrix = _load_transformation_matrix(matrix_path=None)

    # Check if the returned matrix is equivalent to the identity matrix
    assert np.array_equal(result_matrix.rotation, identity_matrix.rotation)
    assert np.array_equal(result_matrix.translation, identity_matrix.translation)


# --- Tests for transform_structure_with_matrix ---


def test_transform_structure_with_matrix_success(
    mocker: MockerFixture,
    mock_structure_coords: Structure,
    non_identity_matrix: TransformationMatrix,
    mock_transformed_structure: Structure,
) -> None:
    """Test successful transformation using a loaded matrix."""
    mock_load = mocker.patch(
        "flatprot.utils.structure_utils._load_transformation_matrix",
        return_value=non_identity_matrix,
    )
    # Mock the transformer class and the parameters class it uses
    mock_transformer_instance = mocker.MagicMock(spec=MatrixTransformer)
    # Mock the transform method that will be called by the lambda in structure_utils
    mock_transformer_instance.transform = mocker.MagicMock(
        return_value=mock_transformed_structure.coordinates
    )  # Needs to return coords

    mock_transformer_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixTransformer",
        return_value=mock_transformer_instance,
    )
    # We also need to patch MatrixTransformParameters as it's now used in structure_utils
    mock_params_instance = mocker.MagicMock(spec=MatrixTransformParameters)
    mock_params_instance.matrix = non_identity_matrix
    mock_params_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixTransformParameters",
        return_value=mock_params_instance,
    )

    # Mock apply_vectorized_transformation to return a new mock structure
    mock_structure_coords.apply_vectorized_transformation.return_value = (
        mock_transformed_structure
    )

    dummy_path = Path("dummy/matrix.npy")
    result_structure = transform_structure_with_matrix(
        mock_structure_coords, dummy_path
    )

    mock_load.assert_called_once_with(dummy_path)

    # Assert MatrixTransformParameters was called correctly inside structure_utils
    mock_params_cls.assert_called_once_with(matrix=non_identity_matrix)

    # Assert MatrixTransformer was initialized correctly with the parameters instance
    mock_transformer_cls.assert_called_once_with(parameters=mock_params_instance)

    # Assert apply_vectorized_transformation was called
    mock_structure_coords.apply_vectorized_transformation.assert_called_once()

    # Check that the lambda passed to apply_vectorized_transformation calls the
    # mock transformer's transform method correctly
    transform_lambda = mock_structure_coords.apply_vectorized_transformation.call_args[
        0
    ][0]
    assert callable(transform_lambda)
    # Call the lambda to trigger the internal call
    _ = transform_lambda(mock_structure_coords.coordinates)
    # Verify the mocked transform method was called by the lambda with correct args
    mock_transformer_instance.transform.assert_called_once_with(
        mock_structure_coords.coordinates, arguments=None
    )

    assert result_structure == mock_transformed_structure


def test_transform_structure_with_matrix_identity(
    mocker: MockerFixture,
    mock_structure_coords: Structure,
    identity_matrix: TransformationMatrix,
    mock_transformed_structure: Structure,
) -> None:
    """Test transformation using the identity matrix (no path or load failure)."""
    mock_load = mocker.patch(
        "flatprot.utils.structure_utils._load_transformation_matrix",
        return_value=identity_matrix,
    )
    # Mock the transformer class and its parameters
    mock_transformer_instance = mocker.MagicMock(spec=MatrixTransformer)
    mock_transformer_instance.transform = mocker.MagicMock(
        return_value=mock_transformed_structure.coordinates
    )
    mock_transformer_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixTransformer",
        return_value=mock_transformer_instance,
    )
    mock_params_instance = mocker.MagicMock(spec=MatrixTransformParameters)
    mock_params_instance.matrix = identity_matrix
    mock_params_cls = mocker.patch(
        "flatprot.utils.structure_utils.MatrixTransformParameters",
        return_value=mock_params_instance,
    )

    mock_structure_coords.apply_vectorized_transformation.return_value = (
        mock_transformed_structure
    )

    # Call with no path, which triggers identity matrix via _load_transformation_matrix
    result_structure = transform_structure_with_matrix(
        mock_structure_coords, matrix_path=None
    )

    mock_load.assert_called_once_with(None)

    # Assert MatrixTransformParameters was called correctly
    mock_params_cls.assert_called_once_with(matrix=identity_matrix)

    # Assert MatrixTransformer was initialized correctly with the parameters instance
    mock_transformer_cls.assert_called_once_with(parameters=mock_params_instance)

    mock_structure_coords.apply_vectorized_transformation.assert_called_once()

    # Check the lambda call
    transform_lambda = mock_structure_coords.apply_vectorized_transformation.call_args[
        0
    ][0]
    assert callable(transform_lambda)
    _ = transform_lambda(mock_structure_coords.coordinates)
    mock_transformer_instance.transform.assert_called_once_with(
        mock_structure_coords.coordinates, arguments=None
    )

    assert result_structure == mock_transformed_structure


def test_transform_structure_with_matrix_no_coords(mocker: MockerFixture) -> None:
    """Test ValueError if structure has no coordinates."""
    mock_struct_no_coords = mocker.MagicMock(spec=Structure)
    # Test the coordinates is None case explicitly checked in the function
    mock_struct_no_coords.coordinates = None
    with pytest.raises(ValueError, match="Structure has no coordinates to transform."):
        transform_structure_with_matrix(mock_struct_no_coords)

    # Test coordinates are empty (should return original structure, not raise error)
    mock_struct_empty_coords = mocker.MagicMock(spec=Structure)
    mock_struct_empty_coords.coordinates = np.empty((0, 3))
    result = transform_structure_with_matrix(mock_struct_empty_coords)
    assert result == mock_struct_empty_coords  # Should log warning and return original


def test_transform_structure_with_matrix_load_error_propagates(
    mocker: MockerFixture, mock_structure_coords: Structure
) -> None:
    """Test that unexpected errors from _load_transformation_matrix propagate."""
    mock_load = mocker.patch(
        "flatprot.utils.structure_utils._load_transformation_matrix",
        side_effect=TransformationError("Failed to load matrix"),
    )
    dummy_path = Path("dummy/matrix.npy")

    with pytest.raises(TransformationError, match="Failed to load matrix"):
        transform_structure_with_matrix(mock_structure_coords, dummy_path)

    mock_load.assert_called_once_with(dummy_path)


# --- Tests for transform_structure_with_inertia ---


def test_transform_structure_with_inertia_success(
    mocker: MockerFixture,
    mock_structure_coords: Structure,
    mock_transformed_structure: Structure,
) -> None:
    """Test successful inertia transformation."""
    mock_transformer_instance = mocker.MagicMock(spec=InertiaTransformer)
    # Mock the transform method itself
    mock_transformer_instance.transform.return_value = np.array(
        [[0.0, 0.0, 0.0]]
    )  # Dummy return

    mock_transformer_cls = mocker.patch(
        "flatprot.utils.structure_utils.InertiaTransformer",
        return_value=mock_transformer_instance,
    )
    mock_default_params_instance = mocker.MagicMock(
        spec=InertiaTransformationParameters
    )
    mock_default_params_cls = mocker.patch(
        "flatprot.utils.structure_utils.InertiaTransformationParameters.default",
        return_value=mock_default_params_instance,
    )

    # Mock apply_vectorized_transformation
    mock_structure_coords.apply_vectorized_transformation.return_value = (
        mock_transformed_structure
    )

    result_structure = transform_structure_with_inertia(mock_structure_coords)

    mock_default_params_cls.assert_called_once()  # Called because no custom params passed
    mock_transformer_cls.assert_called_once()
    # Check that the correct parameters were passed to the constructor
    assert (
        mock_transformer_cls.call_args[1]["parameters"] == mock_default_params_instance
    )

    # Check that apply_vectorized_transformation was called
    mock_structure_coords.apply_vectorized_transformation.assert_called_once()

    # Check that the function passed to apply_vectorized calls the transformer's transform
    # We need to capture the function and call it to verify the internal call
    transform_func = mock_structure_coords.apply_vectorized_transformation.call_args[0][
        0
    ]
    _ = transform_func(mock_structure_coords.coordinates)  # Call the wrapped function

    # Assert InertiaTransformer.transform was called with coords and InertiaTransformationArguments
    mock_transformer_instance.transform.assert_called_once()
    call_args, call_kwargs = mock_transformer_instance.transform.call_args
    assert np.array_equal(call_args[0], mock_structure_coords.coordinates)
    assert isinstance(call_kwargs["arguments"], InertiaTransformationArguments)
    assert call_kwargs["arguments"].residues == mock_structure_coords.residues

    assert result_structure == mock_transformed_structure


def test_transform_structure_with_inertia_custom_params(
    mocker: MockerFixture,
    mock_structure_coords: Structure,
    mock_transformed_structure: Structure,
) -> None:
    """Test inertia transformation with custom parameters."""
    mock_transformer_instance = mocker.MagicMock(spec=InertiaTransformer)
    mock_transformer_instance.transform.return_value = np.array(
        [[0.0, 0.0, 0.0]]
    )  # Dummy return
    mock_transformer_cls = mocker.patch(
        "flatprot.utils.structure_utils.InertiaTransformer",
        return_value=mock_transformer_instance,
    )
    mock_default_params_cls = mocker.patch(
        "flatprot.utils.structure_utils.InertiaTransformationParameters.default"
    )

    mock_structure_coords.apply_vectorized_transformation.return_value = (
        mock_transformed_structure
    )

    # Create custom params - need ResidueType definition or mock
    # Assuming ResidueType can be mocked simply for this test
    mock_residue_type = mocker.MagicMock()
    custom_params = InertiaTransformationParameters(
        residue_weights={mock_residue_type: 1.0}, use_weights=False
    )

    result_structure = transform_structure_with_inertia(
        mock_structure_coords, custom_params
    )

    mock_default_params_cls.assert_not_called()  # Should not be called when custom params are provided
    mock_transformer_cls.assert_called_once()
    # Check that the custom parameters were passed
    assert mock_transformer_cls.call_args[1]["parameters"] == custom_params

    # Check transform call (similar to success case)
    mock_structure_coords.apply_vectorized_transformation.assert_called_once()
    transform_func = mock_structure_coords.apply_vectorized_transformation.call_args[0][
        0
    ]
    _ = transform_func(mock_structure_coords.coordinates)
    mock_transformer_instance.transform.assert_called_once()
    call_args, call_kwargs = mock_transformer_instance.transform.call_args
    assert np.array_equal(call_args[0], mock_structure_coords.coordinates)
    assert isinstance(call_kwargs["arguments"], InertiaTransformationArguments)
    assert call_kwargs["arguments"].residues == mock_structure_coords.residues

    assert result_structure == mock_transformed_structure


def test_transform_structure_with_inertia_no_coords(mocker: MockerFixture) -> None:
    """Test ValueError if structure has no coordinates."""
    mock_struct_no_coords = mocker.MagicMock(spec=Structure)
    mock_struct_no_coords.coordinates = None
    mock_struct_no_coords.residues = [
        mocker.MagicMock()
    ]  # Need residues to pass first check
    with pytest.raises(
        ValueError, match="Structure has no coordinates for inertia transformation."
    ):
        transform_structure_with_inertia(mock_struct_no_coords)

    # Test empty coordinates
    mock_struct_empty_coords = mocker.MagicMock(spec=Structure)
    mock_struct_empty_coords.coordinates = np.empty((0, 3))
    mock_struct_empty_coords.residues = [mocker.MagicMock()]
    result = transform_structure_with_inertia(mock_struct_empty_coords)
    assert result == mock_struct_empty_coords  # Should return original


def test_transform_structure_with_inertia_no_residues(mocker: MockerFixture) -> None:
    """Test ValueError if structure has no residues."""
    # Case 1: residues attribute is an empty list
    mock_struct_empty_residues = mocker.MagicMock(spec=Structure)
    mock_struct_empty_residues.coordinates = np.array([[1.0, 2.0, 3.0]])  # Need coords
    mock_struct_empty_residues.residues = []  # Empty list triggers 'not structure.residues'
    with pytest.raises(
        ValueError, match="Structure has no residues for inertia transformation."
    ):
        transform_structure_with_inertia(mock_struct_empty_residues)

    # Case 2: residues attribute is None
    mock_struct_none_residues = mocker.MagicMock(spec=Structure)
    mock_struct_none_residues.coordinates = np.array([[1.0, 2.0, 3.0]])  # Need coords
    mock_struct_none_residues.residues = (
        None  # None also triggers 'not structure.residues'
    )
    with pytest.raises(
        ValueError, match="Structure has no residues for inertia transformation."
    ):
        transform_structure_with_inertia(mock_struct_none_residues)


def test_transform_structure_with_inertia_transform_error(
    mocker: MockerFixture, mock_structure_coords: Structure
) -> None:
    """Test handling of TransformationError from InertiaTransformer."""
    mock_transformer_instance = mocker.MagicMock(spec=InertiaTransformer)
    # Make the transform method raise the error
    original_error_message = "Inertia calc failed"
    # The TransformationError class adds its own prefix
    raised_error = TransformationError(original_error_message)
    mock_transformer_instance.transform.side_effect = raised_error
    mocker.patch(
        "flatprot.utils.structure_utils.InertiaTransformer",
        return_value=mock_transformer_instance,
    )
    mocker.patch(
        "flatprot.utils.structure_utils.InertiaTransformationParameters.default"
    )

    # Mock apply_vectorized_transformation to actually call the function passed to it
    def call_transform_func(func: Callable[[np.ndarray], np.ndarray]) -> Structure:
        _ = func(mock_structure_coords.coordinates)  # Error should be raised here
        pytest.fail(
            "TransformationError was not raised by transform function"
        )  # Should not be reached

    mock_structure_coords.apply_vectorized_transformation.side_effect = (
        call_transform_func
    )

    # Expect the *original* error message raised by the mock, because the except block re-raises it directly
    expected_error_match = str(
        raised_error
    )  # Match the full message from the original error
    with pytest.raises(TransformationError, match=f"^{expected_error_match}$"):
        transform_structure_with_inertia(mock_structure_coords)

    # Verify transform was called (which raised the error)
    mock_transformer_instance.transform.assert_called_once()


# --- Tests for project_structure_orthographically ---


def test_project_structure_orthographically_success(
    mocker: MockerFixture,
    mock_structure_coords: Structure,
    mock_transformed_structure: Structure,  # Reuse as mock projected structure
) -> None:
    """Test successful orthographic projection."""
    mock_projector_instance = mocker.MagicMock(spec=OrthographicProjection)
    # Mock the project method to return dummy 2D+Depth data
    dummy_projected_coords = np.array(
        [[100.0, 200.0, 5.0], [150.0, 250.0, 10.0]], dtype=np.float32
    )
    mock_projector_instance.project.return_value = dummy_projected_coords

    mock_projector_cls = mocker.patch(
        "flatprot.utils.structure_utils.OrthographicProjection",
        return_value=mock_projector_instance,
    )

    # Mock apply_vectorized_transformation to return a new mock structure
    mock_structure_coords.apply_vectorized_transformation.return_value = (
        mock_transformed_structure
    )
    # Configure the mock projected structure to have the dummy coords
    mock_transformed_structure.coordinates = dummy_projected_coords

    width, height = 800, 600
    result_structure = project_structure_orthographically(
        mock_structure_coords, width=width, height=height
    )

    mock_projector_cls.assert_called_once()  # Assert projector was initialized

    # Assert apply_vectorized_transformation was called
    mock_structure_coords.apply_vectorized_transformation.assert_called_once()

    # Check that the function passed to apply_vectorized calls the projector's project method
    projection_func = mock_structure_coords.apply_vectorized_transformation.call_args[
        0
    ][0]
    proj_result = projection_func(
        mock_structure_coords.coordinates
    )  # Call the wrapped function

    # Assert OrthographicProjection.project was called with correct coords and default parameters
    mock_projector_instance.project.assert_called_once()
    call_args, call_kwargs = mock_projector_instance.project.call_args
    assert np.array_equal(call_args[0], mock_structure_coords.coordinates)
    assert isinstance(call_args[1], OrthographicProjectionParameters)
    assert call_args[1].width == width
    assert call_args[1].height == height
    # Check a few default params were set correctly in the Parameters object
    assert call_args[1].padding_x == 0.05
    assert call_args[1].center is True
    assert np.array_equal(call_args[1].view_direction, np.array([0.0, 0.0, 1.0]))

    # Assert the internal function returned the expected data (and type)
    assert np.array_equal(proj_result, dummy_projected_coords)
    assert proj_result.dtype == np.float32

    # Assert the final returned structure is the one from apply_vectorized_transformation
    assert result_structure == mock_transformed_structure
    # Optionally, check coords on the final structure if not mocking apply_vectorized_transformation deeply
    # assert np.array_equal(result_structure.coordinates, dummy_projected_coords)


def test_project_structure_orthographically_custom_params(
    mocker: MockerFixture,
    mock_structure_coords: Structure,
    mock_transformed_structure: Structure,
) -> None:
    """Test projection with custom parameters (padding, view vectors etc.)."""
    mock_projector_instance = mocker.MagicMock(spec=OrthographicProjection)
    dummy_projected_coords = np.array(
        [[100.0, 200.0, 5.0], [150.0, 250.0, 10.0]], dtype=np.float32
    )
    mock_projector_instance.project.return_value = dummy_projected_coords
    mock_projector_cls = mocker.patch(
        "flatprot.utils.structure_utils.OrthographicProjection",
        return_value=mock_projector_instance,
    )
    mock_structure_coords.apply_vectorized_transformation.return_value = (
        mock_transformed_structure
    )
    mock_transformed_structure.coordinates = dummy_projected_coords

    # Custom parameters
    custom_params = {
        "width": 1000,
        "height": 500,
        "padding_x": 0.1,
        "padding_y": 0.0,
        "maintain_aspect_ratio": False,
        "center_projection": False,
        "view_direction": np.array([1.0, 0.0, 0.0]),
        "up_vector": np.array([0.0, 0.0, 1.0]),
    }

    result_structure = project_structure_orthographically(
        mock_structure_coords, **custom_params
    )

    mock_projector_cls.assert_called_once()
    mock_structure_coords.apply_vectorized_transformation.assert_called_once()

    projection_func = mock_structure_coords.apply_vectorized_transformation.call_args[
        0
    ][0]
    proj_result = projection_func(mock_structure_coords.coordinates)

    # Assert OrthographicProjection.project was called with custom parameters
    mock_projector_instance.project.assert_called_once()
    call_args, call_kwargs = mock_projector_instance.project.call_args
    assert np.array_equal(call_args[0], mock_structure_coords.coordinates)
    assert isinstance(call_args[1], OrthographicProjectionParameters)
    assert call_args[1].width == custom_params["width"]
    assert call_args[1].height == custom_params["height"]
    assert call_args[1].padding_x == custom_params["padding_x"]
    assert call_args[1].padding_y == custom_params["padding_y"]
    assert call_args[1].maintain_aspect_ratio == custom_params["maintain_aspect_ratio"]
    assert np.array_equal(call_args[1].view_direction, custom_params["view_direction"])
    assert np.array_equal(call_args[1].up_vector, custom_params["up_vector"])

    assert np.array_equal(proj_result, dummy_projected_coords)
    assert proj_result.dtype == np.float32
    assert result_structure == mock_transformed_structure


def test_project_structure_orthographically_no_coords(mocker: MockerFixture) -> None:
    """Test ValueError if structure has no coordinates."""
    mock_struct_no_coords = mocker.MagicMock(spec=Structure)
    mock_struct_no_coords.coordinates = None
    with pytest.raises(ValueError, match="Structure has no coordinates to project."):
        project_structure_orthographically(mock_struct_no_coords, width=100, height=100)

    # Test empty coordinates
    mock_struct_empty_coords = mocker.MagicMock(spec=Structure)
    mock_struct_empty_coords.coordinates = np.empty((0, 3))
    result = project_structure_orthographically(
        mock_struct_empty_coords, width=100, height=100
    )
    assert result == mock_struct_empty_coords  # Should return original


def test_project_structure_orthographically_projection_error(
    mocker: MockerFixture, mock_structure_coords: Structure
) -> None:
    """Test handling of ProjectionError from OrthographicProjection."""
    mock_projector_instance = mocker.MagicMock(spec=OrthographicProjection)
    # Make the project method raise the error
    original_error_message = "Projection failed math"
    # The ProjectionError class adds its own prefix
    raised_error = ProjectionError(original_error_message)
    mock_projector_instance.project.side_effect = raised_error

    mocker.patch(
        "flatprot.utils.structure_utils.OrthographicProjection",
        return_value=mock_projector_instance,
    )

    # Mock apply_vectorized_transformation to call the wrapped function
    def call_projection_func(func: Callable[[np.ndarray], np.ndarray]) -> Structure:
        _ = func(mock_structure_coords.coordinates)  # Error should be raised here
        pytest.fail(
            "ProjectionError was not raised by projection function"
        )  # Should not be reached

    mock_structure_coords.apply_vectorized_transformation.side_effect = (
        call_projection_func
    )

    # Expect the *original* error message raised by the mock, because the except block re-raises it directly
    expected_error_match = str(
        raised_error
    )  # Match the full message from the original error
    with pytest.raises(ProjectionError, match=f"^{expected_error_match}$"):
        project_structure_orthographically(mock_structure_coords, width=100, height=100)

    # Verify project was called (which raised the error)
    mock_projector_instance.project.assert_called_once()
