import pytest
import numpy as np

from flatprot.projection.inertia import InertiaProjector, InertiaParameters
from flatprot.projection.structure_elements import (
    InertiaProjector as StructureElementsProjector,
)
from flatprot.projection.structure_elements import StructureElementsParameters
from flatprot.structure.components import Structure, Chain, Residue
from flatprot.structure.secondary import SecondaryStructureType
from flatprot.projection.inertia import InertiaProjectionParameters
from flatprot.projection.structure_elements import StructureElementsProjectionParameters


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


def test_inertia_projector(large_mock_structure):
    projector = InertiaProjector()
    coords = np.vstack([chain.coordinates for chain in large_mock_structure])

    params = InertiaProjectionParameters(
        residues=[
            residue for chain in large_mock_structure for residue in chain.residues
        ]
    )
    projections = projector.project(coords, parameters=params)

    # Calculate variance in original 3D space
    orig_vars = np.var(coords, axis=0)
    sorted_orig_vars = np.sort(orig_vars)[::-1]

    # Calculate variance in projected space
    proj_vars = np.var(projections, axis=0)
    sorted_proj_vars = np.sort(proj_vars)[::-1]

    # The first two dimensions should capture the highest variances
    assert sorted_proj_vars[0] >= 0.9 * sorted_orig_vars[0]
    assert sorted_proj_vars[1] >= 0.9 * sorted_orig_vars[1]
    assert sorted_proj_vars[2] < sorted_proj_vars[1]


def test_inertia_projector_custom_weights():
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    custom_weights = InertiaParameters(
        residue_weights={"ALA": 1.0, "GLY": 0.5}, use_weights=True
    )
    projector = InertiaProjector(parameters=custom_weights)
    params = InertiaProjectionParameters(
        residues=["ALA", "GLY", "ALA"]  # Example residue sequence
    )
    projections = projector.project(coords, parameters=params)

    assert projector.parameters.residue_weights["ALA"] == 1.0
    assert projector.parameters.residue_weights["GLY"] == 0.5


def test_structure_elements_projector(large_mock_structure):
    # Test with different weights for structured regions
    high_weight_params = StructureElementsParameters(
        structure_weight=2.0, non_structure_weight=0.5
    )
    high_weight_projector = StructureElementsProjector(parameters=high_weight_params)
    weighted_projections = high_weight_projector.project(
        large_mock_structure["A"].coordinates,
        StructureElementsProjectionParameters(
            structure_elements=large_mock_structure["A"].secondary_structure
        ),
    )

    # Test with equal weights for comparison
    equal_params = StructureElementsParameters(
        structure_weight=1.0, non_structure_weight=1.0
    )
    equal_projector = StructureElementsProjector(parameters=equal_params)
    equal_projections = equal_projector.project(
        large_mock_structure["A"].coordinates,
        StructureElementsProjectionParameters(
            structure_elements=large_mock_structure["A"].secondary_structure
        ),
    )

    # Focus only on the helical region (first 10 residues)
    helix_coords = large_mock_structure["A"].coordinates[:10]
    weighted_proj_helix = weighted_projections[:10]
    equal_proj_helix = equal_projections[:10]

    # Calculate variance preservation for first two dimensions
    orig_vars = np.var(helix_coords, axis=0)[:2]
    weighted_vars = np.var(weighted_proj_helix, axis=0)[:2]
    equal_vars = np.var(equal_proj_helix, axis=0)[:2]

    weighted_preservation = np.sum(np.abs(weighted_vars - orig_vars))
    equal_preservation = np.sum(np.abs(equal_vars - orig_vars))

    assert weighted_preservation < equal_preservation

    total_weighted_var = np.sum(np.var(weighted_proj_helix, axis=0))
    top2_weighted_var = np.sum(weighted_vars)
    assert top2_weighted_var / total_weighted_var > 0.8


@pytest.mark.parametrize(
    "ProjectorClass", [InertiaProjector, StructureElementsProjector]
)
def test_projector_save_load(tmp_path, ProjectorClass):
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    save_path = tmp_path / "projection.npz"

    projector = ProjectorClass()
    params = (
        InertiaProjectionParameters(residues=["ALA"] * len(coords))
        if ProjectorClass == InertiaProjector
        else None
    )
    original_projections = projector.project(coords, parameters=params)
    projector.save(save_path)

    new_projector = ProjectorClass()
    new_projector.load(save_path)
    loaded_projections = new_projector.project(coords, parameters=params)

    assert np.allclose(original_projections, loaded_projections)
