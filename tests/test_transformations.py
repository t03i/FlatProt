import pytest
import numpy as np

from flatprot.transformation.inertia import (
    InertiaTransformer,
    InertiaTransformParameters,
    InertiaTransformerParameters,
)
from flatprot.transformation.structure_elements import (
    StructureElementsTransformer,
    StructureElementsTransformParameters,
    StructureElementsTransformerParameters,
)
from flatprot.transformation.matrix import MatrixTransformer, MatrixTransformParameters
from flatprot.core.components import Structure, Chain, Residue
from flatprot.core.secondary import SecondaryStructureType
from flatprot.transformation.utils import TransformationMatrix


@pytest.fixture
def mock_structure():
    # Chain A: simple helix-like structure
    coords_a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    residues_a = [Residue.ALA for i in range(len(coords_a))]
    index_a = np.arange(len(coords_a)) + 1
    chain_a = Chain("A", residues_a, index_a, coords_a)
    chain_a.add_secondary_structure(SecondaryStructureType.HELIX, 1, 4)

    # Chain B: non-structured region
    coords_b = np.array([[0, 0, 2], [1, 1, 3], [2, 2, 4]])
    residues_b = [Residue.GLY for i in range(len(coords_b))]
    index_b = np.arange(len(coords_b)) + 1
    chain_b = Chain("B", residues_b, index_b, coords_b)

    # Create structure with both chains
    return Structure([chain_a, chain_b])


@pytest.fixture
def large_mock_structure():
    # Chain A: larger helix-like structure
    coords_a = np.array([[i, np.sin(i / 3), np.cos(i / 3)] for i in range(20)])
    residues_a = [Residue.ALA for _ in range(len(coords_a))]
    index_a = np.arange(len(coords_a)) + 1
    chain_a = Chain("A", residues_a, index_a, coords_a)
    chain_a.add_secondary_structure(SecondaryStructureType.HELIX, 1, 10)

    return Structure([chain_a])


def test_matrix_transformer_comprehensive():
    # Create test transformation
    rotation = np.array(
        [
            [0, -1, 0],  # 90-degree rotation around z-axis
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    translation = np.array([1.0, 2.0, 3.0])
    transformation = TransformationMatrix(rotation=rotation, translation=translation)

    # Create test coordinates
    coords = np.array(
        [
            [1.0, 0.0, 0.0],  # Point on x-axis
            [0.0, 1.0, 0.0],  # Point on y-axis
            [1.0, 1.0, 1.0],  # Point in general position
        ]
    )

    # Test basic transformation
    transformer = MatrixTransformer()
    transformed = transformer.transform(
        coords, parameters=MatrixTransformParameters(matrix=transformation)
    )

    # Expected results after transformation:
    # 1. First rotate around z-axis by 90 degrees
    # 2. Then translate by [1,2,3]
    expected = np.array(
        [
            [1.0, 3.0, 3.0],
            [0.0, 2.0, 3.0],
            [0.0, 3.0, 4.0],
        ]
    )

    assert np.allclose(transformed, expected)

    # Test that cached transformation is used
    assert transformer._cached_transformation is not None
    assert np.allclose(transformer._cached_transformation.rotation, rotation)
    assert np.allclose(transformer._cached_transformation.translation, translation)


def test_inertia_transformer(large_mock_structure):
    transformer = InertiaTransformer()
    coords = np.vstack([chain.coordinates for chain in large_mock_structure])

    params = InertiaTransformParameters(
        residues=[
            residue for chain in large_mock_structure for residue in chain.residues
        ]
    )
    transformed = transformer.transform(coords, parameters=params)

    # Calculate variance in original 3D space
    orig_vars = np.var(coords, axis=0)
    sorted_orig_vars = np.sort(orig_vars)[::-1]

    # Calculate variance in projected space
    proj_vars = np.var(transformed, axis=0)
    sorted_proj_vars = np.sort(proj_vars)[::-1]

    # The first two dimensions should capture the highest variances
    assert sorted_proj_vars[0] >= 0.9 * sorted_orig_vars[0]
    assert sorted_proj_vars[1] >= 0.9 * sorted_orig_vars[1]
    assert sorted_proj_vars[2] < sorted_proj_vars[1]


def test_inertia_transformer_custom_weights():
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    custom_weights = InertiaTransformerParameters(
        residue_weights={"ALA": 1.0, "GLY": 0.5}, use_weights=True
    )
    transformer = InertiaTransformer(parameters=custom_weights)
    params = InertiaTransformParameters(
        residues=["ALA", "GLY", "ALA"]  # Example residue sequence
    )
    transformed = transformer.transform(coords, parameters=params)

    assert transformer.parameters.residue_weights["ALA"] == 1.0
    assert transformer.parameters.residue_weights["GLY"] == 0.5


def test_structure_elements_transformer(large_mock_structure):
    # Test with different weights for structured regions
    high_weight_params = StructureElementsTransformerParameters(
        structure_weight=2.0, non_structure_weight=0.5
    )
    high_weight_transformer = StructureElementsTransformer(
        parameters=high_weight_params
    )
    weighted_transformed = high_weight_transformer.transform(
        large_mock_structure["A"].coordinates,
        StructureElementsTransformParameters(
            structure_elements=large_mock_structure["A"].secondary_structure
        ),
    )

    # Test with equal weights for comparison
    equal_params = StructureElementsTransformerParameters(
        structure_weight=1.0, non_structure_weight=1.0
    )
    equal_transformer = StructureElementsTransformer(parameters=equal_params)
    equal_transformed = equal_transformer.transform(
        large_mock_structure["A"].coordinates,
        StructureElementsTransformParameters(
            structure_elements=large_mock_structure["A"].secondary_structure
        ),
    )

    # Focus only on the helical region (first 10 residues)
    helix_coords = large_mock_structure["A"].coordinates[:10]
    weighted_proj_helix = weighted_transformed[:10]
    equal_proj_helix = equal_transformed[:10]

    # Calculate variance preservation for first two dimensions
    orig_vars = np.var(helix_coords, axis=0)[:2]
    weighted_vars = np.var(weighted_proj_helix, axis=0)[:2]
    equal_vars = np.var(equal_proj_helix, axis=0)[:2]

    weighted_preservation = np.sum(np.abs(weighted_vars - orig_vars))
    equal_preservation = np.sum(np.abs(equal_vars - orig_vars))

    assert weighted_preservation > equal_preservation

    total_weighted_var = np.sum(np.var(weighted_proj_helix, axis=0))
    top2_weighted_var = np.sum(weighted_vars)
    assert top2_weighted_var / total_weighted_var > 0.8


@pytest.mark.parametrize(
    "TransformerClass",
    [
        InertiaTransformer,
        StructureElementsTransformer,
        MatrixTransformer,
    ],
)
def test_transformer_save_load(tmp_path, TransformerClass):
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    save_path = tmp_path / "transformation.npz"

    transformer = TransformerClass()
    if TransformerClass == MatrixTransformer:
        params = MatrixTransformParameters(
            matrix=TransformationMatrix(rotation=np.eye(3), translation=np.zeros(3))
        )
    elif TransformerClass == InertiaTransformer:
        params = InertiaTransformParameters(residues=["ALA"] * len(coords))
    else:
        params = None

    original_transformed = transformer.transform(coords, parameters=params)
    transformer.save(save_path)

    new_transformer = TransformerClass()
    new_transformer.load(save_path)
    loaded_transformed = new_transformer.transform(coords, parameters=params)

    assert np.allclose(original_transformed, loaded_transformed)
