# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from flatprot.core.structure import Chain, Structure
from flatprot.core.types import ResidueType, SecondaryStructureType
from flatprot.core.coordinates import ResidueCoordinate, ResidueRange


# --- Fixtures ---


@pytest.fixture
def sample_chain_a() -> Chain:
    """Provides a simple, continuous sample chain A."""
    return Chain(
        chain_id="A",
        residues=[ResidueType.ALA] * 5,  # 5 residues: ALA
        index=np.array([10, 11, 12, 13, 14]),  # Indices 10 to 14
        coordinates=np.random.rand(5, 3),  # Random coordinates
    )


@pytest.fixture
def sample_chain_b_discontinuous() -> Chain:
    """Provides a sample chain B with a discontinuity in indices."""
    return Chain(
        chain_id="B",
        residues=[ResidueType.GLY] * 3,  # 3 residues: GLY
        index=np.array([5, 6, 9]),  # Indices 5, 6, then jumps to 9
        coordinates=np.random.rand(3, 3),
    )


@pytest.fixture
def sample_structure(
    sample_chain_a: Chain, sample_chain_b_discontinuous: Chain
) -> Structure:
    """Provides a sample structure containing chain A and chain B."""
    # Provide an explicit ID for testing
    return Structure([sample_chain_a, sample_chain_b_discontinuous], id="test_struct")


# --- Chain Tests ---


def test_chain_init_success(sample_chain_a: Chain) -> None:
    """Test successful Chain initialization."""
    assert sample_chain_a.id == "A"
    assert len(sample_chain_a) == 5
    assert sample_chain_a.num_residues == 5
    assert len(sample_chain_a.coordinates) == 5
    assert len(sample_chain_a.residues) == 5
    assert np.array_equal(sample_chain_a.index, np.array([10, 11, 12, 13, 14]))


def test_chain_init_length_mismatch() -> None:
    """Test Chain initialization fails with mismatched input lengths."""
    with pytest.raises(ValueError, match="must have the same length"):
        Chain(
            chain_id="C",
            residues=[ResidueType.ALA] * 3,
            index=np.array([1, 2, 3, 4]),  # Mismatched length
            coordinates=np.random.rand(3, 3),
        )
    with pytest.raises(ValueError, match="must have the same length"):
        Chain(
            chain_id="C",
            residues=[ResidueType.ALA] * 3,
            index=np.array([1, 2, 3]),
            coordinates=np.random.rand(4, 3),  # Mismatched length
        )


def test_chain_iteration(sample_chain_a: Chain) -> None:
    """Test iterating over a Chain yields ResidueCoordinate objects."""
    coords = list(sample_chain_a)
    assert len(coords) == 5
    assert all(isinstance(c, ResidueCoordinate) for c in coords)
    assert coords[0] == ResidueCoordinate("A", 10, ResidueType.ALA, 0)
    assert coords[-1] == ResidueCoordinate("A", 14, ResidueType.ALA, 4)


def test_chain_getitem(sample_chain_a: Chain) -> None:
    """Test accessing residues by index using __getitem__."""
    coord = sample_chain_a[12]
    assert isinstance(coord, ResidueCoordinate)
    assert coord.chain_id == "A"
    assert coord.residue_index == 12
    assert coord.residue_type == ResidueType.ALA
    assert coord.coordinate_index == 2  # 12 is the 3rd element (0-indexed)

    with pytest.raises(KeyError):
        _ = sample_chain_a[15]  # Index not present


def test_chain_contains(sample_chain_a: Chain) -> None:
    """Test checking residue index existence using __contains__."""
    assert 11 in sample_chain_a
    assert 14 in sample_chain_a
    assert 9 not in sample_chain_a
    assert 15 not in sample_chain_a


def test_chain_coordinate_index(sample_chain_a: Chain) -> None:
    """Test retrieving the internal coordinate index for a residue index."""
    assert sample_chain_a.coordinate_index(10) == 0
    assert sample_chain_a.coordinate_index(14) == 4
    with pytest.raises(KeyError):
        sample_chain_a.coordinate_index(9)


def test_chain_str(sample_chain_a: Chain) -> None:
    """Test the string representation of a Chain."""
    assert str(sample_chain_a) == "Chain A"


def test_chain_to_ranges_continuous(sample_chain_a: Chain) -> None:
    """Test converting a continuous chain to a single ResidueRange."""
    ranges = sample_chain_a.to_ranges()
    assert len(ranges) == 1
    expected_range = ResidueRange("A", 10, 14, 0)
    assert ranges[0] == expected_range


def test_chain_to_ranges_discontinuous(
    sample_chain_b_discontinuous: Chain,
) -> None:
    """Test converting a discontinuous chain to multiple ResidueRanges."""
    ranges = sample_chain_b_discontinuous.to_ranges()
    assert len(ranges) == 2
    expected_ranges = [
        ResidueRange("B", 5, 6, 0),  # First contiguous segment
        ResidueRange("B", 9, 9, 2),  # Second segment after the jump
    ]
    assert ranges == expected_ranges


def test_chain_to_ranges_empty() -> None:
    """Test converting an empty chain to an empty list of ranges."""
    empty_chain = Chain("E", [], np.array([]), np.empty((0, 3)))
    assert empty_chain.to_ranges() == []


# --- Secondary Structure Tests ---


def test_chain_secondary_structure_default_coil(sample_chain_a: Chain) -> None:
    """Test that secondary_structure defaults to COIL for all residues if none are added."""
    ss = sample_chain_a.secondary_structure
    assert len(ss) == 1
    expected_range = ResidueRange("A", 10, 14, 0, SecondaryStructureType.COIL)
    assert ss[0] == expected_range


def test_chain_add_secondary_structure_success(sample_chain_a: Chain) -> None:
    """Test successfully adding a valid secondary structure element."""
    sample_chain_a.add_secondary_structure(SecondaryStructureType.HELIX, 11, 13)

    ss = sample_chain_a.secondary_structure
    # Expected: Coil(10-10), Helix(11-13), Coil(14-14)
    expected_ss = [
        ResidueRange("A", 10, 10, 0, SecondaryStructureType.COIL),
        ResidueRange("A", 11, 13, 1, SecondaryStructureType.HELIX),
        ResidueRange("A", 14, 14, 4, SecondaryStructureType.COIL),
    ]
    assert ss == expected_ss


def test_chain_add_secondary_structure_invalid_index(sample_chain_a: Chain) -> None:
    """Test adding secondary structure with non-existent residue indices raises ValueError."""
    with pytest.raises(ValueError, match="not found in chain"):
        sample_chain_a.add_secondary_structure(
            SecondaryStructureType.SHEET, 9, 12
        )  # 9 not in chain
    with pytest.raises(ValueError, match="not found in chain"):
        sample_chain_a.add_secondary_structure(
            SecondaryStructureType.SHEET, 11, 15
        )  # 15 not in chain


def test_chain_secondary_structure_add_overlapping(sample_chain_a: Chain) -> None:
    """Test behavior when adding overlapping secondary structures.

    The expectation is that the definitions from the structure added *later* alphabetically
    (based on range sorting within the property) will take precedence in the overlapping region.
    """
    # Add Helix first (start=10), then Sheet (start=11)
    sample_chain_a.add_secondary_structure(SecondaryStructureType.HELIX, 10, 12)
    sample_chain_a.add_secondary_structure(SecondaryStructureType.SHEET, 11, 13)

    ss = sample_chain_a.secondary_structure

    # Based on internal sorting (Helix 10-12 comes before Sheet 11-13):
    # The map becomes {10: H, 11: H, 12: H}, then {11: S, 12: S, 13: S}
    # Resulting map: {10: H, 11: S, 12: S, 13: S}
    # Expected segments: H(10-10), S(11-13), Coil(14-14)
    expected_ss = [
        ResidueRange("A", 10, 10, 0, SecondaryStructureType.HELIX),
        ResidueRange("A", 11, 13, 1, SecondaryStructureType.SHEET),
        ResidueRange("A", 14, 14, 4, SecondaryStructureType.COIL),
    ]
    assert ss == expected_ss


def test_chain_secondary_structure_add_discontinuous_error(
    sample_chain_b_discontinuous: Chain,
) -> None:
    """Test adding secondary structure spanning a discontinuity raises ValueError."""
    # Chain B has indices 5, 6, 9. Gap between 6 and 9.
    # Attempt to add structure from 6 to 9.
    with pytest.raises(ValueError, match="not contiguous"):
        sample_chain_b_discontinuous.add_secondary_structure(
            SecondaryStructureType.HELIX, 6, 9
        )


def test_chain_secondary_structure_discontinuous_chain(
    sample_chain_b_discontinuous: Chain,
) -> None:
    """Test secondary structure generation on a discontinuous chain.

    Add structure before the gap and verify the output segments correctly.
    """
    # Chain B has indices 5, 6, 9. Add Helix to 5-6.
    sample_chain_b_discontinuous.add_secondary_structure(
        SecondaryStructureType.HELIX, 5, 6
    )

    ss = sample_chain_b_discontinuous.secondary_structure

    # Expected: Helix segment for 5-6, separate Coil segment for 9.
    expected_ss = [
        ResidueRange("B", 5, 6, 0, SecondaryStructureType.HELIX),
        ResidueRange("B", 9, 9, 2, SecondaryStructureType.COIL),
    ]
    assert ss == expected_ss


# --- Transformation Tests ---
# Simple transformer function for testing
def translate_coords(coords: np.ndarray, offset: float = 1.0) -> np.ndarray:
    """Simple translation function for testing."""
    return coords + offset


def test_chain_apply_vectorized_transformation(sample_chain_a: Chain) -> None:
    """Test applying a transformation to a single chain."""
    original_coords = sample_chain_a.coordinates.copy()

    new_chain = sample_chain_a.apply_vectorized_transformation(
        lambda coords: translate_coords(coords, 2.5)
    )

    # Verify new chain has correct ID and transformed coordinates
    assert new_chain.id == sample_chain_a.id
    assert new_chain.num_residues == sample_chain_a.num_residues
    np.testing.assert_allclose(new_chain.coordinates, original_coords + 2.5)

    # Verify original chain is unchanged
    np.testing.assert_allclose(sample_chain_a.coordinates, original_coords)

    # Verify metadata is preserved
    assert new_chain.residues == sample_chain_a.residues
    assert np.array_equal(new_chain.index, sample_chain_a.index)
    # Add SS to original and check if it's copied (shallow copy expected)
    sample_chain_a.add_secondary_structure(SecondaryStructureType.HELIX, 11, 12)
    _ = sample_chain_a.secondary_structure  # Force generation/caching if any

    new_chain_transformed_again = sample_chain_a.apply_vectorized_transformation(
        lambda coords: translate_coords(coords, 2.5)
    )
    assert len(new_chain_transformed_again._Chain__secondary_structure) == 1
    assert (
        new_chain_transformed_again._Chain__secondary_structure[
            0
        ].secondary_structure_type
        == SecondaryStructureType.HELIX
    )


def test_chain_apply_vectorized_transformation_shape_error(
    sample_chain_a: Chain,
) -> None:
    """Test that transformation changing coordinate shape raises ValueError."""
    with pytest.raises(ValueError, match="changed coordinate array shape"):
        sample_chain_a.apply_vectorized_transformation(lambda coords: coords[:, :2])


# --- Structure Tests ---


def test_structure_init(sample_structure: Structure) -> None:
    """Test Structure initialization and basic properties."""
    assert len(sample_structure) == 2
    assert "A" in sample_structure
    assert "B" in sample_structure
    assert "C" not in sample_structure


def test_structure_getitem(sample_structure: Structure, sample_chain_a: Chain) -> None:
    """Test accessing chains by ID using __getitem__."""
    assert sample_structure["A"] is sample_chain_a
    with pytest.raises(KeyError):
        _ = sample_structure["C"]


def test_structure_contains_coordinate(sample_structure: Structure) -> None:
    """Test __contains__ with ResidueCoordinate objects."""
    assert ResidueCoordinate("A", 10) in sample_structure
    assert ResidueCoordinate("B", 9) in sample_structure
    assert (
        ResidueCoordinate("A", 15) not in sample_structure
    )  # Valid chain, invalid index
    assert ResidueCoordinate("C", 1) not in sample_structure  # Invalid chain
    assert (
        ResidueCoordinate("B", 7) not in sample_structure
    )  # Valid chain, index in discontinuity


def test_structure_iteration(sample_structure: Structure) -> None:
    """Test iterating over a Structure yields chain ID, Chain object tuples."""
    items = list(sample_structure)
    assert len(items) == 2
    assert all(isinstance(item[0], str) for item in items)
    assert all(isinstance(item[1], Chain) for item in items)
    assert set(item[0] for item in items) == {"A", "B"}


def test_structure_items(sample_structure: Structure) -> None:
    """Test iterating over structure items."""
    items = list(sample_structure.items())
    assert len(items) == 2
    assert set(item[0] for item in items) == {"A", "B"}
    assert all(isinstance(item[1], Chain) for item in items)


def test_structure_values(sample_structure: Structure) -> None:
    """Test iterating over structure values."""
    values = list(sample_structure.values())
    assert len(values) == 2
    assert all(isinstance(v, Chain) for v in values)
    assert set(v.id for v in values) == {"A", "B"}


def test_structure_residues(sample_structure: Structure) -> None:
    """Test getting concatenated residues from all chains."""
    residues = sample_structure.residues
    # Chain A has 5 ALAs, Chain B has 3 GLYs
    assert len(residues) == 8
    assert residues.count(ResidueType.ALA) == 5
    assert residues.count(ResidueType.GLY) == 3


def test_structure_coordinates(sample_structure: Structure) -> None:
    """Test getting concatenated coordinates from all chains."""
    coords = sample_structure.coordinates
    # Chain A has 5 residues, Chain B has 3
    assert coords.shape == (8, 3)
    # Check concatenation order (assuming structure preserves input order)
    np.testing.assert_array_equal(coords[:5], sample_structure["A"].coordinates)
    np.testing.assert_array_equal(coords[5:], sample_structure["B"].coordinates)


def test_structure_str(sample_structure: Structure) -> None:
    """Test the string representation of a Structure."""
    # Update assertion to match new format including the ID
    assert str(sample_structure) == "Structure(ID: test_struct, Chains: ['A', 'B'])"


def test_structure_apply_vectorized_transformation(
    sample_structure: Structure,
) -> None:
    """Test applying a transformation to the entire structure."""
    original_coords_struct = sample_structure.coordinates.copy()
    original_coords_a = sample_structure["A"].coordinates.copy()
    original_coords_b = sample_structure["B"].coordinates.copy()

    new_structure = sample_structure.apply_vectorized_transformation(
        lambda coords: translate_coords(coords, -1.5)
    )

    # Verify new structure has correct number of chains and transformed coords
    assert len(new_structure) == len(sample_structure)
    assert "A" in new_structure and "B" in new_structure
    np.testing.assert_allclose(new_structure.coordinates, original_coords_struct - 1.5)

    # Verify individual chains within the new structure have correct coords
    np.testing.assert_allclose(new_structure["A"].coordinates, original_coords_a - 1.5)
    np.testing.assert_allclose(new_structure["B"].coordinates, original_coords_b - 1.5)

    # Verify original structure and its chains are unchanged
    np.testing.assert_allclose(sample_structure.coordinates, original_coords_struct)
    np.testing.assert_allclose(sample_structure["A"].coordinates, original_coords_a)
    np.testing.assert_allclose(sample_structure["B"].coordinates, original_coords_b)


def test_structure_apply_vectorized_transformation_shape_error(
    sample_structure: Structure,
) -> None:
    """Test structure transformation fails if function changes total coordinate shape."""
    with pytest.raises(ValueError, match="changed coordinate array shape"):
        sample_structure.apply_vectorized_transformation(lambda coords: coords[:-1, :])
