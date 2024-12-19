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


def test_inertia_projector_structure_scope(mock_structure):
    projector = InertiaProjector(scope=ProjectionScope.STRUCTURE)
    projections = projector.project(mock_structure)

    assert len(projections) == 2
    assert "A" in projections
    assert "B" in projections
    assert projections["A"].shape == (4, 3)  # 4 residues, 2D coordinates
    assert projections["B"].shape == (3, 3)  # 3 residues, 2D coordinates


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


def test_structure_elements_projector(mock_structure):
    params = StructureElementsParameters(structure_weight=2.0, non_structure_weight=0.5)
    projector = StructureElementsProjector(parameters=params)
    projections = projector.project(mock_structure)

    assert len(projections) == 2
    assert projections["A"].shape == (4, 3)
    assert projections["B"].shape == (3, 3)


def test_structure_elements_weights(mock_structure):
    # Test that structured regions get higher weights
    params = StructureElementsParameters(structure_weight=2.0, non_structure_weight=0.5)
    projector = StructureElementsProjector(parameters=params)

    # Chain A has a helix (structured)
    # Chain B has no structure
    projections_structure = projector.project(mock_structure)

    # Create a projector with equal weights for comparison
    equal_params = StructureElementsParameters(
        structure_weight=1.0, non_structure_weight=1.0
    )
    projector_equal = StructureElementsProjector(parameters=equal_params)
    projections_equal = projector_equal.project(mock_structure)

    # The projections should be different due to different weights
    assert not np.allclose(projections_structure["A"], projections_equal["A"])


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
