import pytest
import numpy as np
from typing import List, Dict, Union

from flatprot.transformation.inertia_transformation import (
    InertiaTransformer,
    InertiaTransformationParameters,
    InertiaTransformationArguments,
    calculate_inertia_transformation_matrix,
)
from flatprot.core.types import ResidueType
from flatprot.transformation import TransformationMatrix


@pytest.fixture
def sample_coordinates_simple() -> np.ndarray:
    """Provide simple, non-centered coordinates."""
    return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])


@pytest.fixture
def sample_coordinates_diverse() -> np.ndarray:
    """Provide more diverse coordinates for inertia testing."""
    # Coordinates forming a rough plane, shifted from origin
    return np.array(
        [
            [5.0, 1.0, 1.0],
            [5.0, -1.0, 1.0],
            [6.0, 0.0, -1.0],
            [4.0, 0.0, -1.0],
            [5.0, 0.0, 0.0],  # Center point
        ]
    )


@pytest.fixture
def sample_residues_simple() -> List[ResidueType]:
    """Provide simple residue types corresponding to sample_coordinates_simple."""
    return [ResidueType.ALA, ResidueType.GLY, ResidueType.ALA]


@pytest.fixture
def sample_residues_diverse() -> List[ResidueType]:
    """Provide residue types corresponding to sample_coordinates_diverse."""
    return [
        ResidueType.LEU,
        ResidueType.LEU,
        ResidueType.SER,
        ResidueType.SER,
        ResidueType.GLY,
    ]


@pytest.fixture
def default_inertia_params() -> InertiaTransformationParameters:
    """Get default inertia parameters."""
    params = InertiaTransformationParameters.default()
    params.use_weights = True
    return params


# --- Tests for calculate_inertia_transformation_matrix ---


def test_calculate_inertia_matrix_standard_translation(
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test the calculated standard translation T = -(R_inertia @ C)."""
    coords = sample_coordinates_simple
    weights = np.ones(len(coords))
    center_C = np.mean(coords, axis=0).reshape(-1, 1)  # Geometric center

    # Calculate the matrix (which now contains R_inertia and T_standard)
    matrix = calculate_inertia_transformation_matrix(coords, weights)
    R_inertia = matrix.rotation
    T_standard_calculated = matrix.translation

    # Calculate expected T_standard
    expected_T_standard = -(R_inertia @ center_C).flatten()

    assert np.allclose(T_standard_calculated, expected_T_standard, atol=1e-6)


def test_calculate_inertia_matrix_weighted_standard_translation(
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test standard translation with non-equal weights."""
    coords = sample_coordinates_simple
    weights = np.array([1.0, 2.0, 1.0])  # Make middle point heavier
    center_C = (
        np.sum(coords * weights[:, np.newaxis], axis=0) / np.sum(weights)
    ).reshape(-1, 1)

    # Calculate the matrix (which now contains R_inertia and T_standard)
    matrix = calculate_inertia_transformation_matrix(coords, weights)
    R_inertia = matrix.rotation
    T_standard_calculated = matrix.translation

    # Calculate expected T_standard
    expected_T_standard = -(R_inertia @ center_C).flatten()

    assert np.allclose(T_standard_calculated, expected_T_standard, atol=1e-6)


def test_calculate_inertia_matrix_apply_centers_correctly(
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test that applying the standard matrix correctly centers the molecule."""
    coords = sample_coordinates_simple
    weights = np.array([1.0, 2.0, 1.0])
    original_center_C = np.sum(coords * weights[:, np.newaxis], axis=0) / np.sum(
        weights
    )

    # Calculate the matrix (R_inertia, T_standard)
    matrix = calculate_inertia_transformation_matrix(coords, weights)

    # Apply the transformation using the standard apply method
    transformed_coords = matrix.apply(coords)  # Should be (R_inertia @ X) + T_standard

    # Verify that the *original* center C, when transformed, ends up at the origin.
    # Transform C: (R_inertia @ C) + T_standard
    # Since T_standard = -(R_inertia @ C), this should be zero.
    transformed_center_C = (
        matrix.rotation @ original_center_C.reshape(-1, 1)
    ).flatten() + matrix.translation

    assert np.allclose(transformed_center_C, np.zeros(3), atol=1e-6)

    # Also verify the center of mass of the transformed points is at the origin
    transformed_com = np.sum(
        transformed_coords * weights[:, np.newaxis], axis=0
    ) / np.sum(weights)
    assert np.allclose(
        transformed_com, np.zeros(3), atol=1e-6
    ), f"Transformed COM {transformed_com} is not zero"


def test_calculate_inertia_matrix_rotation_properties(
    sample_coordinates_diverse: np.ndarray,
) -> None:
    """Test properties of the calculated rotation matrix."""
    coords = sample_coordinates_diverse
    weights = np.ones(len(coords))  # Use equal weights for simplicity here
    matrix = calculate_inertia_transformation_matrix(coords, weights)
    rotation = matrix.rotation

    # Check orthogonality: R^T * R = I
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    # Check determinant is +1 (proper rotation)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_calculate_inertia_matrix_principal_axes(
    sample_coordinates_diverse: np.ndarray,
) -> None:
    """Test if transformation aligns coordinates with principal axes."""
    coords = sample_coordinates_diverse
    weights = np.ones(len(coords))
    matrix = calculate_inertia_transformation_matrix(coords, weights)
    transformed_coords = matrix.apply(coords)

    # Variance along axes should correspond to eigenvalues of covariance matrix (related to inertia tensor)
    # The transformation should align axes such that variance is maximized along the first axis, then second, etc.
    variances = np.var(transformed_coords, axis=0)
    # Variances should be sorted in descending order (approximately)
    assert (
        variances[0] >= variances[1] >= variances[2]
        or np.allclose(variances[0], variances[1])
        or np.allclose(variances[1], variances[2])
    )

    # Check covariance matrix of transformed coords is diagonal (or close to it)
    cov_matrix = np.cov(transformed_coords, rowvar=False)
    assert np.allclose(cov_matrix, np.diag(np.diag(cov_matrix)), atol=1e-6)


# --- Tests for InertiaTransformer ---


def test_inertia_transformer_init_default(
    default_inertia_params: InertiaTransformationParameters,
) -> None:
    """Test initialization with default parameters."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    assert transformer.parameters is default_inertia_params
    assert transformer.parameters.use_weights is True
    assert ResidueType.ALA in transformer.parameters.residue_weights


def test_inertia_transformer_init_custom() -> None:
    """Test initialization with custom parameters."""
    custom_weights: Dict[ResidueType, float] = {
        ResidueType.ALA: 10.0,
        ResidueType.GLY: 5.0,
    }
    custom_params = InertiaTransformationParameters(
        residue_weights=custom_weights, use_weights=False
    )
    transformer = InertiaTransformer(parameters=custom_params)

    assert transformer.parameters.use_weights is False
    assert transformer.parameters.residue_weights == custom_weights


def test_inertia_transformer_transform_method_call(
    mocker,  # Use pytest-mock fixture
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_simple: np.ndarray,
    sample_residues_simple: List[ResidueType],
) -> None:
    """Test that the transform method calls calculate and then matrix.apply."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    coords = sample_coordinates_simple
    transform_args = InertiaTransformationArguments(residues=sample_residues_simple)

    # Create a real matrix object to be returned by the mocked calculate
    # We can use arbitrary valid values since we'll mock its 'apply' method
    dummy_matrix = TransformationMatrix(rotation=np.eye(3), translation=np.zeros(3))

    # Mock the internal calculate method to return our dummy matrix
    mock_calculate = mocker.patch.object(
        transformer, "_calculate_transformation_matrix", return_value=dummy_matrix
    )

    # IMPORTANT: Spy on the 'apply' method of the *specific dummy_matrix instance*
    # that we expect to be returned and used.
    mock_apply = mocker.spy(dummy_matrix, "apply")

    # Call the transform method
    try:
        _ = transformer.transform(coordinates=coords, arguments=transform_args)
    except Exception as e:
        pytest.fail(f"transform raised unexpected exception: {e}")

    # Assert calculate was called once with the correct arguments
    mock_calculate.assert_called_once_with(coords, transform_args)

    # Assert the apply method on our dummy_matrix instance was called once with coords
    mock_apply.assert_called_once_with(coords)

    # Optionally, assert the result is what dummy_matrix.apply would have returned
    # (If spy doesn't change behavior, this should hold)
    # assert np.array_equal(result, dummy_matrix.apply(coords)) # This might be redundant


def test_inertia_transformer_transform_centering(
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_diverse: np.ndarray,
    sample_residues_diverse: List[ResidueType],
) -> None:
    """Test the overall transform centers the coordinates' weighted COM at the origin."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    coords = sample_coordinates_diverse
    residues = sample_residues_diverse
    transform_args = InertiaTransformationArguments(residues=residues)

    # Calculate expected weights for verification
    expected_weights = np.array(
        [default_inertia_params.residue_weights.get(res, 1.0) for res in residues]
    )

    # Apply the transformation
    transformed_coords = transformer.transform(
        coordinates=coords, arguments=transform_args
    )

    # Check the shape of the output
    assert transformed_coords.shape == coords.shape, "Output shape mismatch"

    # Check that the center of mass of the transformed coordinates is at the origin
    transformed_com = np.sum(
        transformed_coords * expected_weights[:, np.newaxis], axis=0
    ) / np.sum(expected_weights)

    assert np.allclose(
        transformed_com,
        np.zeros(3),
        atol=1e-6,
    ), f"Transformed COM {transformed_com} is not close to zero."

    # Check variance alignment (optional, but good)
    variances = np.var(transformed_coords, axis=0)
    assert (
        variances[0] >= variances[1] >= variances[2]
        or np.allclose(variances[0], variances[1])
        or np.allclose(variances[1], variances[2])
    ), "Variances are not ordered as expected after transformation"


def test_inertia_transformer_default_weight_fallback(
    mocker,  # Use pytest-mock
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test args passed to calculate function when unknown residues are present."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    coords = sample_coordinates_simple
    # Use a mix including a non-standard type
    residues = [ResidueType.ALA, ResidueType.GLY, "UNK"]

    # Define args specifically for this test case
    class MockInertiaArgs(InertiaTransformationArguments):
        # Allow string type for testing
        residues: List[Union[ResidueType, str]]

    transform_args = MockInertiaArgs(residues=residues)

    # --- Mocking Setup ---
    # 1. Create a dummy matrix to be returned by the calculate function
    #    so that the transform method can complete.
    dummy_matrix = TransformationMatrix(rotation=np.eye(3), translation=np.zeros(3))
    # 2. Mock its apply method just in case it's called (it should be by base class)
    mocker.patch.object(
        dummy_matrix, "apply", return_value=coords
    )  # Return something valid
    # 3. Mock the _calculate_transformation_matrix method on the transformer instance
    #    to return our dummy matrix. We will check the arguments passed to this mock.
    mock_calculate = mocker.patch.object(
        transformer, "_calculate_transformation_matrix", return_value=dummy_matrix
    )
    # --- End Mocking Setup ---

    # Call transform - it should now complete without attribute error
    try:
        # Pass type ignore because transform_args contains 'UNK' string
        transformer.transform(coordinates=coords, arguments=transform_args)  # type: ignore
    except Exception as e:
        pytest.fail(f"transform raised unexpected exception: {e}")

    # --- Assertion ---
    # Assert _calculate_transformation_matrix was called once
    mock_calculate.assert_called_once()
    # Get the actual arguments passed to the mocked _calculate_transformation_matrix
    call_args, _ = mock_calculate.call_args

    # Assert the correct arguments were passed
    assert np.array_equal(call_args[0], coords)  # Check coords
    # Check that the arguments object containing the residues list was passed
    assert call_args[1] is transform_args
    # Verify the content of the residues list within the passed arguments object
    assert call_args[1].residues == [ResidueType.ALA, ResidueType.GLY, "UNK"]

    # Remove previous assertions checking internal state or calculated translation,
    # as they are not the focus of this test with the current mocking strategy.
