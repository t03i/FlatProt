import pytest
import numpy as np

from flatprot.projection.projector import ProjectionScope
from flatprot.projection.inertia import InertiaProjector, InertiaParameters
from flatprot.projection.structure_elements import (
    InertiaProjector as StructureElementsProjector,
)
from flatprot.projection.structure_elements import StructureElementsParameters
from flatprot.structure.components import Structure, Chain, Residue
from flatprot.structure.secondary import SecondaryStructureType


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


def test_inertia_projector_structure_scope(large_mock_structure):
    projector = InertiaProjector(scope=ProjectionScope.STRUCTURE)
    projections = projector.project(large_mock_structure)

    # Basic shape checks
    assert len(projections) == 1
    assert "A" in projections

    # Calculate variance in original 3D space
    all_coords = np.vstack([chain.coordinates for chain in large_mock_structure])
    orig_vars = np.var(all_coords, axis=0)
    sorted_orig_vars = np.sort(orig_vars)[::-1]  # Sort in descending order

    # Calculate variance in projected space
    all_proj = np.vstack([projections[chain_id] for chain_id in projections])
    proj_vars = np.var(all_proj, axis=0)
    sorted_proj_vars = np.sort(proj_vars)[::-1]  # Sort in descending order

    # The first two dimensions should capture the highest variances
    assert sorted_proj_vars[0] >= 0.9 * sorted_orig_vars[0]
    assert sorted_proj_vars[1] >= 0.9 * sorted_orig_vars[1]
    assert (
        sorted_proj_vars[2] < sorted_proj_vars[1]
    )  # Third dimension should have less variance


def test_inertia_projector_chain_scope(mock_structure):
    projector = InertiaProjector(scope=ProjectionScope.CHAIN)
    projections = projector.project(mock_structure)

    assert len(projections) == 2
    # Each chain should have different projections
    proj_a = projections["A"]
    proj_b = projections["B"]
    assert not np.allclose(proj_a[:2], proj_b[:2])


def test_inertia_projector_custom_weights():
    # Test with custom residue weights
    custom_weights = InertiaParameters(
        residue_weights={"ALA": 1.0, "GLY": 0.5}, use_weights=True
    )
    projector = InertiaProjector(parameters=custom_weights)
    assert projector.parameters.residue_weights["ALA"] == 1.0
    assert projector.parameters.residue_weights["GLY"] == 0.5


def test_structure_elements_projector(large_mock_structure):
    # Test with different weights for structured regions
    high_weight_params = StructureElementsParameters(
        structure_weight=2.0, non_structure_weight=0.5
    )
    high_weight_projector = StructureElementsProjector(parameters=high_weight_params)
    weighted_projections = high_weight_projector.project(large_mock_structure)

    # Test with equal weights for comparison
    equal_params = StructureElementsParameters(
        structure_weight=1.0, non_structure_weight=1.0
    )
    equal_projector = StructureElementsProjector(parameters=equal_params)
    equal_projections = equal_projector.project(large_mock_structure)

    # Focus only on the helical region (first 10 residues)
    helix_coords = large_mock_structure["A"].coordinates[:10]
    weighted_proj_helix = weighted_projections["A"][:10]
    equal_proj_helix = equal_projections["A"][:10]

    # Calculate variance preservation for first two dimensions
    orig_vars = np.var(helix_coords, axis=0)[:2]  # Only first two dimensions
    weighted_vars = np.var(weighted_proj_helix, axis=0)[:2]
    equal_vars = np.var(equal_proj_helix, axis=0)[:2]

    # Calculate how well variance is preserved relative to original
    weighted_preservation = np.sum(np.abs(weighted_vars - orig_vars))
    equal_preservation = np.sum(np.abs(equal_vars - orig_vars))

    # The weighted projection should preserve variance better in first two dimensions
    # for the helical region specifically
    assert weighted_preservation < equal_preservation

    # Also verify that the first two dimensions capture most of the variance
    # for the helical region in the weighted projection
    total_weighted_var = np.sum(np.var(weighted_proj_helix, axis=0))
    top2_weighted_var = np.sum(weighted_vars)
    assert (
        top2_weighted_var / total_weighted_var > 0.8
    )  # At least 80% of variance in top 2 dimensions


@pytest.mark.parametrize(
    "ProjectorClass", [InertiaProjector, StructureElementsProjector]
)
def test_projector_save_load(tmp_path, ProjectorClass, mock_structure):
    # Test saving and loading projections
    save_path = tmp_path / "projection.npz"

    projector = ProjectorClass()
    original_projections = projector.project(mock_structure)
    projector.save(save_path)

    # Create new projector and load saved projection
    new_projector = ProjectorClass()
    new_projector.load(save_path)
    loaded_projections = new_projector.project(mock_structure)

    # Projections should be identical
    for chain_id in original_projections:
        assert np.allclose(original_projections[chain_id], loaded_projections[chain_id])
