import pytest
import numpy as np
from typing import Tuple

from flatprot.transformation.matrix_transformation import (
    MatrixTransformer,
    MatrixTransformParameters,
)
from flatprot.transformation.transformation_matrix import TransformationMatrix


@pytest.fixture
def sample_coordinates() -> np.ndarray:
    """Provide sample coordinates for testing."""
    return np.array(
        [
            [1.0, 0.0, 0.0],  # Point on x-axis
            [0.0, 1.0, 0.0],  # Point on y-axis
            [1.0, 1.0, 1.0],  # Point in general position
        ]
    )


@pytest.fixture
def sample_transformation() -> Tuple[TransformationMatrix, np.ndarray]:
    """
    Provide a sample transformation matrix and expected transformed coordinates
    using the standard (R @ X) + T convention.
    """
    rotation = np.array(
        [
            [0, -1, 0],  # 90-degree rotation around z-axis
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    translation = np.array([1.0, 2.0, 3.0])
    transformation_matrix = TransformationMatrix(
        rotation=rotation, translation=translation
    )

    # Original coordinates:
    # X = [[1, 0, 0],
    #      [0, 1, 0],
    #      [1, 1, 1]]

    # Expected results after standard transformation: (R @ X.T).T + T
    # Point 1: r = [1, 0, 0]
    #   R @ r = [0, 1, 0]
    #   (R @ r) + T = [0+1, 1+2, 0+3] = [1, 3, 3]
    # Point 2: r = [0, 1, 0]
    #   R @ r = [-1, 0, 0]
    #   (R @ r) + T = [-1+1, 0+2, 0+3] = [0, 2, 3]
    # Point 3: r = [1, 1, 1]
    #   R @ r = [-1, 1, 1]
    #   (R @ r) + T = [-1+1, 1+2, 1+3] = [0, 3, 4]
    expected_coords = np.array(
        [
            [1.0, 3.0, 3.0],
            [0.0, 2.0, 3.0],
            [0.0, 3.0, 4.0],
        ]
    )

    return transformation_matrix, expected_coords


def test_matrix_transformer_initialization(
    sample_transformation: Tuple[TransformationMatrix, np.ndarray],
) -> None:
    """Test if the MatrixTransformer initializes correctly."""
    transformation_matrix, _ = sample_transformation
    parameters = MatrixTransformParameters(matrix=transformation_matrix)
    transformer = MatrixTransformer(parameters=parameters)

    assert transformer is not None
    assert transformer._cached_transformation is transformation_matrix
    assert np.allclose(
        transformer._cached_transformation.rotation, transformation_matrix.rotation
    )
    assert np.allclose(
        transformer._cached_transformation.translation,
        transformation_matrix.translation,
    )


def test_matrix_transformer_transform(
    sample_coordinates: np.ndarray,
    sample_transformation: Tuple[TransformationMatrix, np.ndarray],
) -> None:
    """Test the coordinate transformation using the MatrixTransformer."""
    transformation_matrix, expected_coords = sample_transformation
    coords = sample_coordinates

    parameters = MatrixTransformParameters(matrix=transformation_matrix)
    transformer = MatrixTransformer(parameters=parameters)

    # Transform coordinates
    # The `transform` method in BaseTransformation expects coordinates and optional args/kwargs
    # which are passed to _calculate_transformation_matrix and then apply.
    # MatrixTransformer._calculate_transformation_matrix ignores coordinates and returns the cached matrix.
    # BaseTransformation.apply uses the result of _calculate_transformation_matrix.
    transformed_coords = transformer.transform(
        coordinates=coords, arguments=None
    )  # No specific arguments needed here

    assert transformed_coords.shape == expected_coords.shape
    assert np.allclose(transformed_coords, expected_coords)


def test_matrix_transformer_cached_matrix(
    sample_coordinates: np.ndarray,
    sample_transformation: Tuple[TransformationMatrix, np.ndarray],
) -> None:
    """Test that the cached transformation matrix is used."""
    transformation_matrix, _ = sample_transformation
    coords = sample_coordinates

    parameters = MatrixTransformParameters(matrix=transformation_matrix)
    transformer = MatrixTransformer(parameters=parameters)

    # Access the cached transformation before calling transform
    cached_before = transformer._cached_transformation

    # Call transform
    _ = transformer.transform(coordinates=coords, arguments=None)

    # Access the cached transformation after calling transform
    cached_after = transformer._cached_transformation

    # Ensure the cached matrix is the same one initially provided and didn't change
    assert cached_before is transformation_matrix
    assert cached_after is transformation_matrix
    assert np.allclose(cached_after.rotation, transformation_matrix.rotation)
    assert np.allclose(cached_after.translation, transformation_matrix.translation)
