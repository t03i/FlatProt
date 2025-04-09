import pytest
import numpy as np
from typing import List, Dict

from flatprot.transformation.inertia_transformation import (
    InertiaTransformer,
    InertiaTransformationParameters,
    InertiaTransformationArguments,
    calculate_inertia_transformation_matrix,
)
from flatprot.core.types import ResidueType


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
    return InertiaTransformationParameters.default()


# --- Tests for calculate_inertia_transformation_matrix ---


def test_calculate_inertia_matrix_geometric_center(
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test translation uses geometric center when weights are equal."""
    coords = sample_coordinates_simple
    weights = np.ones(len(coords))
    matrix = calculate_inertia_transformation_matrix(coords, weights)

    expected_translation = np.mean(coords, axis=0)
    assert np.allclose(matrix.translation, expected_translation)

    # Check transformed center is origin
    transformed_coords = matrix.apply(coords)
    assert np.allclose(np.mean(transformed_coords, axis=0), np.zeros(3), atol=1e-6)


def test_calculate_inertia_matrix_center_of_mass(
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test translation uses center of mass for non-equal weights."""
    coords = sample_coordinates_simple
    weights = np.array([1.0, 2.0, 1.0])  # Make middle point heavier
    matrix = calculate_inertia_transformation_matrix(coords, weights)

    expected_translation = np.sum(coords * weights[:, np.newaxis], axis=0) / np.sum(
        weights
    )
    assert np.allclose(matrix.translation, expected_translation)

    # Check transformed weighted center is origin
    transformed_coords = matrix.apply(coords)
    transformed_center = np.sum(
        transformed_coords * weights[:, np.newaxis], axis=0
    ) / np.sum(weights)
    assert np.allclose(transformed_center, np.zeros(3), atol=1e-6)


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


def test_inertia_transformer_calculate_matrix_no_weights(
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_simple: np.ndarray,
    sample_residues_simple: List[ResidueType],
) -> None:
    """Test _calculate_transformation_matrix with use_weights=False."""
    params_no_weight = InertiaTransformationParameters(
        residue_weights=default_inertia_params.residue_weights, use_weights=False
    )
    transformer = InertiaTransformer(parameters=params_no_weight)
    coords = sample_coordinates_simple
    # Create arguments needed by _calculate_transformation_matrix via transform
    transform_args = InertiaTransformationArguments(residues=sample_residues_simple)

    # Use the public transform method, which calls the internal _calculate
    transformer.transform(coordinates=coords, arguments=transform_args)
    # Get the calculated matrix
    calculated_matrix = transformer._cached_transformation
    assert calculated_matrix is not None

    # Expected matrix uses geometric center
    expected_translation = np.mean(coords, axis=0)
    assert np.allclose(calculated_matrix.translation, expected_translation)


def test_inertia_transformer_calculate_matrix_with_weights(
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_simple: np.ndarray,
    sample_residues_simple: List[ResidueType],
) -> None:
    """Test _calculate_transformation_matrix with use_weights=True."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    coords = sample_coordinates_simple
    residues = sample_residues_simple
    transform_args = InertiaTransformationArguments(residues=residues)

    transformer.transform(coordinates=coords, arguments=transform_args)
    calculated_matrix = transformer._cached_transformation
    assert calculated_matrix is not None

    # Expected matrix uses center of mass based on default residue weights
    weights = np.array(
        [
            default_inertia_params.residue_weights[ResidueType.ALA],
            default_inertia_params.residue_weights[ResidueType.GLY],
            default_inertia_params.residue_weights[ResidueType.ALA],
        ]
    )
    expected_translation = np.sum(coords * weights[:, np.newaxis], axis=0) / np.sum(
        weights
    )
    assert np.allclose(calculated_matrix.translation, expected_translation)


def test_inertia_transformer_transform_centering(
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_diverse: np.ndarray,
    sample_residues_diverse: List[ResidueType],
) -> None:
    """Test the transform centers the coordinates' weighted COM at the origin."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    coords = sample_coordinates_diverse
    residues = sample_residues_diverse
    transform_args = InertiaTransformationArguments(residues=residues)

    # --- Recalculate expected weights and COM (T) for verification ---
    expected_weights = np.array(
        [default_inertia_params.residue_weights.get(res, 1.0) for res in residues]
    )
    assert np.sum(expected_weights) > 0, "Sum of weights should be positive"
    expected_translation = np.sum(
        coords * expected_weights[:, np.newaxis], axis=0
    ) / np.sum(expected_weights)
    # --- End Recalculation ---

    transformed_coords = transformer.transform(
        coordinates=coords, arguments=transform_args
    )

    # --- Verification Steps ---
    # 1. Check the calculated translation vector used internally matches expectation
    calculated_matrix = transformer._cached_transformation
    assert calculated_matrix is not None, "Transformation matrix was not cached"
    assert np.allclose(
        calculated_matrix.translation, expected_translation
    ), "Internal translation vector mismatch"

    # 2. Check the shape of the output
    assert transformed_coords.shape == coords.shape, "Output shape mismatch"

    # 3. Check that the center of mass of the transformed coordinates is at the origin
    # Use the same weights derived earlier
    transformed_com = np.sum(
        transformed_coords * expected_weights[:, np.newaxis], axis=0
    ) / np.sum(expected_weights)

    assert np.allclose(
        transformed_com,
        np.zeros(3),
        atol=1e-6,  # Use standard tolerance first
    ), f"Transformed COM {transformed_com} is not close to zero."

    # 4. Check variance alignment (optional, but good to keep)
    # This confirms the rotation part is orienting correctly.
    variances = np.var(transformed_coords, axis=0)
    assert (
        variances[0] >= variances[1] >= variances[2]
        or np.allclose(variances[0], variances[1])
        or np.allclose(variances[1], variances[2])
    ), "Variances are not ordered as expected after transformation"


def test_inertia_transformer_default_weight_fallback(
    default_inertia_params: InertiaTransformationParameters,
    sample_coordinates_simple: np.ndarray,
) -> None:
    """Test that unknown residues get a default weight of 1.0."""
    transformer = InertiaTransformer(parameters=default_inertia_params)
    coords = sample_coordinates_simple
    # Use a mix including a non-standard type (using UNK or a placeholder)
    # We need to handle non-enum types if they can appear
    residues = [ResidueType.ALA, ResidueType.GLY, "UNK"]  # type: ignore

    # Define args specifically for this test case
    class MockInertiaArgs(InertiaTransformationArguments):
        residues: List[ResidueType | str]

    transform_args = MockInertiaArgs(residues=residues)

    # Temporarily allow non-standard residue types in parameters for testing fallback
    try:
        transformer.transform(coordinates=coords, arguments=transform_args)  # type: ignore
        calculated_matrix = transformer._cached_transformation
        assert calculated_matrix is not None

        # Expected weights: ALA, GLY, UNK (1.0)
        weights = np.array(
            [
                default_inertia_params.residue_weights[ResidueType.ALA],
                default_inertia_params.residue_weights[ResidueType.GLY],
                1.0,  # Default fallback weight
            ]
        )
        expected_translation = np.sum(coords * weights[:, np.newaxis], axis=0) / np.sum(
            weights
        )
        assert np.allclose(calculated_matrix.translation, expected_translation)
    finally:
        # Restore original parameters if modified, although InertiaTransformer doesn't modify in-place
        pass
