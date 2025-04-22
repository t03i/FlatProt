import pytest

from flatprot.core.coordinates import (
    ResidueCoordinate,
    ResidueRange,
    ResidueRangeSet,
)
from flatprot.core.types import ResidueType, SecondaryStructureType


# --- Tests for ResidueCoordinate ---


def test_residue_coordinate_creation() -> None:
    """Test basic creation of ResidueCoordinate."""
    coord = ResidueCoordinate("A", 10)
    assert coord.chain_id == "A"
    assert coord.residue_index == 10
    assert coord.residue_type is None
    assert coord.coordinate_index is None


def test_residue_coordinate_creation_with_optional() -> None:
    """Test creation with optional attributes."""
    coord = ResidueCoordinate("B", 5, ResidueType.ALA, 100)
    assert coord.chain_id == "B"
    assert coord.residue_index == 5
    assert coord.residue_type == ResidueType.ALA
    assert coord.coordinate_index == 100


def test_residue_coordinate_from_string() -> None:
    """Test creating ResidueCoordinate from string."""
    coord = ResidueCoordinate.from_string("C:25")
    assert coord.chain_id == "C"
    assert coord.residue_index == 25
    assert coord.residue_type is None
    assert coord.coordinate_index is None


def test_residue_coordinate_from_string_invalid() -> None:
    """Test invalid string format for ResidueCoordinate."""
    with pytest.raises(ValueError):
        ResidueCoordinate.from_string("A-10")
    with pytest.raises(ValueError):
        ResidueCoordinate.from_string("A:")
    with pytest.raises(ValueError):
        ResidueCoordinate.from_string(":10")


def test_residue_coordinate_str() -> None:
    """Test string representation of ResidueCoordinate."""
    coord1 = ResidueCoordinate("A", 10)
    assert str(coord1) == "A:10"
    coord2 = ResidueCoordinate("B", 5, ResidueType.GLY, 99)
    assert str(coord2) == "B:5 (GLY)"


def test_residue_coordinate_eq() -> None:
    """Test equality comparison of ResidueCoordinate."""
    coord1 = ResidueCoordinate("A", 10)
    coord2 = ResidueCoordinate("A", 10)
    coord3 = ResidueCoordinate("A", 11)
    coord4 = ResidueCoordinate("B", 10)
    coord5 = ResidueCoordinate("A", 10, ResidueType.ALA)  # Type shouldn't affect eq
    assert coord1 == coord2
    assert coord1 == ResidueCoordinate.from_string("A:10")
    assert coord1 == coord5
    assert coord1 != coord3
    assert coord1 != coord4

    with pytest.raises(TypeError):
        _ = coord1 == "A:10"


def test_residue_coordinate_lt() -> None:
    """Test less than comparison of ResidueCoordinate."""
    coord1 = ResidueCoordinate("A", 10)
    coord2 = ResidueCoordinate("A", 11)
    coord3 = ResidueCoordinate("B", 5)
    coord4 = ResidueCoordinate("A", 10)

    assert coord1 < coord2
    assert coord1 < coord3  #
    assert not (coord1 < coord4)
    assert not (coord2 < coord1)
    assert not (coord3 < coord1)

    with pytest.raises(TypeError):
        _ = coord1 < "A:11"


# --- Tests for ResidueRange ---


@pytest.fixture
def residue_range_a() -> ResidueRange:
    """Fixture for a sample ResidueRange."""
    return ResidueRange("A", 10, 20)


@pytest.fixture
def residue_range_b() -> ResidueRange:
    """Fixture for another sample ResidueRange."""
    return ResidueRange(
        "B",
        5,
        15,
        coordinates_start_index=50,
        secondary_structure_type=SecondaryStructureType.HELIX,
    )


def test_residue_range_creation(residue_range_a: ResidueRange) -> None:
    """Test basic creation of ResidueRange."""
    assert residue_range_a.chain_id == "A"
    assert residue_range_a.start == 10
    assert residue_range_a.end == 20
    assert residue_range_a.coordinates_start_index is None
    assert residue_range_a.secondary_structure_type is None


def test_residue_range_creation_with_optional(residue_range_b: ResidueRange) -> None:
    """Test creation with optional attributes."""
    assert residue_range_b.chain_id == "B"
    assert residue_range_b.start == 5
    assert residue_range_b.end == 15
    assert residue_range_b.coordinates_start_index == 50
    assert residue_range_b.secondary_structure_type == SecondaryStructureType.HELIX


def test_residue_range_invalid_creation() -> None:
    """Test creation with invalid start/end."""
    with pytest.raises(ValueError, match="Invalid range: 20 > 10"):
        ResidueRange("A", 20, 10)


def test_residue_range_iteration(residue_range_a: ResidueRange) -> None:
    """Test iterating over ResidueCoordinates in a range."""
    expected_coords = [ResidueCoordinate("A", i) for i in range(10, 21)]
    assert list(residue_range_a) == expected_coords


def test_residue_range_len(residue_range_a: ResidueRange) -> None:
    """Test length calculation of ResidueRange."""
    assert len(residue_range_a) == 11


def test_residue_range_from_string() -> None:
    """Test creating ResidueRange from string."""
    range_ = ResidueRange.from_string("C:5-12")
    assert range_.chain_id == "C"
    assert range_.start == 5
    assert range_.end == 12


def test_residue_range_from_string_invalid() -> None:
    """Test invalid string formats for ResidueRange."""
    with pytest.raises(ValueError):
        ResidueRange.from_string("A:10")  # Missing dash
    with pytest.raises(ValueError):
        ResidueRange.from_string("A:10-")
    with pytest.raises(ValueError):
        ResidueRange.from_string("A:-15")
    with pytest.raises(ValueError):
        ResidueRange.from_string("A10-15")  # Missing colon
    with pytest.raises(ValueError):
        ResidueRange.from_string(":10-15")
    with pytest.raises(ValueError):
        ResidueRange.from_string("A:10-b")  # Non-integer


def test_residue_range_str(
    residue_range_a: ResidueRange, residue_range_b: ResidueRange
) -> None:
    """Test string representation of ResidueRange."""
    assert str(residue_range_a) == "A:10-20"
    assert str(residue_range_b) == "B:5-15 (HELIX)"


def test_residue_range_lt() -> None:
    """Test less than comparison of ResidueRange."""
    r1 = ResidueRange("A", 10, 20)
    r2 = ResidueRange("A", 15, 25)
    r3 = ResidueRange("B", 5, 15)
    r4 = ResidueRange("A", 10, 15)  # Same start, different end
    r5 = ResidueRange("A", 10, 20)  # Equal

    assert r1 < r2
    assert r1 < r3
    assert r4 < r2
    assert not (r1 < r4)  # Only start matters for <
    assert not (r1 < r5)
    assert not (r2 < r1)
    assert not (r3 < r1)

    with pytest.raises(TypeError):
        _ = r1 < "A:15-25"


def test_residue_range_eq(residue_range_a: ResidueRange) -> None:
    """Test equality comparison of ResidueRange."""
    r1 = ResidueRange("A", 10, 20)
    r2 = ResidueRange("A", 10, 21)
    r3 = ResidueRange("B", 10, 20)
    r4 = ResidueRange(
        "A", 10, 20, coordinates_start_index=0
    )  # Optional args don't affect eq
    assert residue_range_a == r1
    assert residue_range_a == ResidueRange.from_string("A:10-20")
    assert residue_range_a == r4
    assert residue_range_a != r2
    assert residue_range_a != r3

    with pytest.raises(TypeError):
        _ = residue_range_a == "A:10-20"  # Different type


def test_residue_range_contains_coordinate(residue_range_a: ResidueRange) -> None:
    """Test if a range contains a ResidueCoordinate."""
    inside = ResidueCoordinate("A", 15)
    edge_start = ResidueCoordinate("A", 10)
    edge_end = ResidueCoordinate("A", 20)
    outside_before = ResidueCoordinate("A", 9)
    outside_after = ResidueCoordinate("A", 21)
    wrong_chain = ResidueCoordinate("B", 15)

    assert inside in residue_range_a
    assert edge_start in residue_range_a
    assert edge_end in residue_range_a
    assert outside_before not in residue_range_a
    assert outside_after not in residue_range_a
    assert wrong_chain not in residue_range_a


def test_residue_range_contains_range(residue_range_a: ResidueRange) -> None:
    """Test if a range contains another ResidueRange."""
    fully_inside = ResidueRange("A", 12, 18)
    identical = ResidueRange("A", 10, 20)
    touching_start = ResidueRange("A", 10, 15)
    touching_end = ResidueRange("A", 15, 20)
    overlapping_start = ResidueRange("A", 5, 15)
    overlapping_end = ResidueRange("A", 15, 25)
    outside = ResidueRange("A", 30, 40)
    wrong_chain = ResidueRange("B", 12, 18)

    assert fully_inside in residue_range_a
    assert identical in residue_range_a
    assert touching_start in residue_range_a
    assert touching_end in residue_range_a
    assert overlapping_start not in residue_range_a
    assert overlapping_end not in residue_range_a
    assert outside not in residue_range_a
    assert wrong_chain not in residue_range_a


def test_residue_range_to_set(residue_range_a: ResidueRange) -> None:
    """Test converting a ResidueRange to a ResidueRangeSet."""
    range_set = residue_range_a.to_set()
    assert isinstance(range_set, ResidueRangeSet)
    assert len(range_set.ranges) == 1
    assert range_set.ranges[0] == residue_range_a


# --- Tests for ResidueRange.is_adjacent_to ---


def test_residue_range_is_adjacent_true() -> None:
    """Test that adjacent ranges return True."""
    r1 = ResidueRange.from_string("A:10-15")
    r2 = ResidueRange.from_string("A:16-20")
    assert r1.is_adjacent_to(r2)
    assert r2.is_adjacent_to(r1)


def test_residue_range_is_adjacent_false_gap() -> None:
    """Test that non-adjacent ranges with a gap return False."""
    r1 = ResidueRange.from_string("A:10-15")
    r2 = ResidueRange.from_string("A:17-20")  # Gap at 16
    assert not r1.is_adjacent_to(r2)
    assert not r2.is_adjacent_to(r1)


def test_residue_range_is_adjacent_false_overlap() -> None:
    """Test that overlapping ranges return False."""
    r1 = ResidueRange.from_string("A:10-15")
    r2 = ResidueRange.from_string("A:14-20")  # Overlap at 14, 15
    assert not r1.is_adjacent_to(r2)
    assert not r2.is_adjacent_to(r1)


def test_residue_range_is_adjacent_false_touching() -> None:
    """Test that touching ranges (end == start) return False."""
    r1 = ResidueRange.from_string("A:10-15")
    r2 = ResidueRange.from_string("A:15-20")  # Touching at 15
    assert not r1.is_adjacent_to(r2)
    assert not r2.is_adjacent_to(r1)


def test_residue_range_is_adjacent_false_different_chain() -> None:
    """Test that ranges in different chains return False."""
    r1 = ResidueRange.from_string("A:10-15")
    r2 = ResidueRange.from_string("B:16-20")  # Different chains
    assert not r1.is_adjacent_to(r2)
    assert not r2.is_adjacent_to(r1)


def test_residue_range_is_adjacent_type_error() -> None:
    """Test that comparing with a non-ResidueRange raises TypeError."""
    r1 = ResidueRange.from_string("A:10-15")
    with pytest.raises(TypeError):
        r1.is_adjacent_to("A:16-20")  # type: ignore


def test_residue_range_is_adjacent_to_coordinate_true() -> None:
    """Test adjacent range and coordinate."""
    r1 = ResidueRange.from_string("A:10-15")
    coord_before = ResidueCoordinate.from_string("A:9")
    coord_after = ResidueCoordinate.from_string("A:16")
    assert r1.is_adjacent_to(coord_before)
    assert r1.is_adjacent_to(coord_after)
    # Symmetry is not defined for range/coordinate, but test implementation
    # assert coord_before.is_adjacent_to(r1) # Requires Coordinate.is_adjacent_to
    # assert coord_after.is_adjacent_to(r1) # Requires Coordinate.is_adjacent_to


def test_residue_range_is_adjacent_to_coordinate_false() -> None:
    """Test non-adjacent range and coordinate."""
    r1 = ResidueRange.from_string("A:10-15")
    coord_inside = ResidueCoordinate.from_string("A:12")
    coord_touch_start = ResidueCoordinate.from_string("A:10")
    coord_touch_end = ResidueCoordinate.from_string("A:15")
    coord_gap_before = ResidueCoordinate.from_string("A:8")
    coord_gap_after = ResidueCoordinate.from_string("A:17")
    coord_diff_chain = ResidueCoordinate.from_string("B:9")
    coord_diff_chain_adj = ResidueCoordinate.from_string("B:16")

    assert not r1.is_adjacent_to(coord_inside)
    assert not r1.is_adjacent_to(coord_touch_start)
    assert not r1.is_adjacent_to(coord_touch_end)
    assert not r1.is_adjacent_to(coord_gap_before)
    assert not r1.is_adjacent_to(coord_gap_after)
    assert not r1.is_adjacent_to(coord_diff_chain)
    assert not r1.is_adjacent_to(coord_diff_chain_adj)


# --- Tests for ResidueRangeSet ---


@pytest.fixture
def residue_range_set_simple() -> ResidueRangeSet:
    """Fixture for a simple ResidueRangeSet."""
    r1 = ResidueRange("A", 10, 20)
    r2 = ResidueRange("A", 30, 35)
    return ResidueRangeSet([r2, r1])  # Test sorting on init


@pytest.fixture
def residue_range_set_multi_chain() -> ResidueRangeSet:
    """Fixture for a ResidueRangeSet with multiple chains."""
    r1 = ResidueRange("B", 5, 15)
    r2 = ResidueRange("A", 50, 60)
    r3 = ResidueRange("A", 10, 20)
    return ResidueRangeSet([r1, r2, r3])  # Test sorting


def test_residue_range_set_creation(residue_range_set_simple: ResidueRangeSet) -> None:
    """Test creation and sorting of ResidueRangeSet."""
    assert len(residue_range_set_simple.ranges) == 2
    # Check if sorted correctly
    assert residue_range_set_simple.ranges[0] == ResidueRange("A", 10, 20)
    assert residue_range_set_simple.ranges[1] == ResidueRange("A", 30, 35)


def test_residue_range_set_creation_multi_chain(
    residue_range_set_multi_chain: ResidueRangeSet,
) -> None:
    """Test creation and sorting with multiple chains."""
    assert len(residue_range_set_multi_chain.ranges) == 3
    # Check sorting (Chain A first, then by start index, then Chain B)
    assert residue_range_set_multi_chain.ranges[0] == ResidueRange("A", 10, 20)
    assert residue_range_set_multi_chain.ranges[1] == ResidueRange("A", 50, 60)
    assert residue_range_set_multi_chain.ranges[2] == ResidueRange("B", 5, 15)


def test_residue_range_set_from_string() -> None:
    """Test creating ResidueRangeSet from string."""
    set_ = ResidueRangeSet.from_string("A:10-20, B:5-15 ,A:50-60")
    assert len(set_.ranges) == 3
    # Check sorting after parsing
    assert set_.ranges[0] == ResidueRange("A", 10, 20)
    assert set_.ranges[1] == ResidueRange("A", 50, 60)
    assert set_.ranges[2] == ResidueRange("B", 5, 15)


def test_residue_range_set_from_string_single() -> None:
    """Test creating ResidueRangeSet from string with one range."""
    set_ = ResidueRangeSet.from_string("C:1-5")
    assert len(set_.ranges) == 1
    assert set_.ranges[0] == ResidueRange("C", 1, 5)


def test_residue_range_set_from_string_empty() -> None:
    """Test creating ResidueRangeSet from an empty string (should fail)."""
    # This depends on ResidueRange.from_string behavior for empty parts
    with pytest.raises(ValueError):  # Assumes ResidueRange.from_string fails on ""
        ResidueRangeSet.from_string("")
    # Test with just a comma
    with pytest.raises(ValueError):  # Assumes ResidueRange.from_string fails on ""
        ResidueRangeSet.from_string(",")


def test_residue_range_set_iteration(
    residue_range_set_simple: ResidueRangeSet,
) -> None:
    """Test iterating over all ResidueCoordinates in a set."""
    expected_coords = [ResidueCoordinate("A", i) for i in range(10, 21)] + [
        ResidueCoordinate("A", i) for i in range(30, 36)
    ]
    assert list(residue_range_set_simple) == expected_coords


def test_residue_range_set_len(residue_range_set_simple: ResidueRangeSet) -> None:
    """Test total length calculation of ResidueRangeSet."""
    len_r1 = 20 - 10 + 1
    len_r2 = 35 - 30 + 1
    assert len(residue_range_set_simple) == len_r1 + len_r2


def test_residue_range_set_str(
    residue_range_set_multi_chain: ResidueRangeSet,
) -> None:
    """Test string representation of ResidueRangeSet."""
    # Ranges are sorted in the __init__
    assert str(residue_range_set_multi_chain) == "A:10-20,A:50-60,B:5-15"


def test_residue_range_set_contains_coordinate(
    residue_range_set_multi_chain: ResidueRangeSet,
) -> None:
    """Test if a set contains a ResidueCoordinate."""
    coord_a1 = ResidueCoordinate("A", 15)
    coord_a2 = ResidueCoordinate("A", 55)
    coord_b1 = ResidueCoordinate("B", 10)
    coord_a_out = ResidueCoordinate("A", 5)
    coord_b_out = ResidueCoordinate("B", 20)
    coord_c = ResidueCoordinate("C", 1)

    assert coord_a1 in residue_range_set_multi_chain
    assert coord_a2 in residue_range_set_multi_chain
    assert coord_b1 in residue_range_set_multi_chain
    assert coord_a_out not in residue_range_set_multi_chain
    assert coord_b_out not in residue_range_set_multi_chain
    assert coord_c not in residue_range_set_multi_chain


def test_residue_range_set_contains_range(
    residue_range_set_multi_chain: ResidueRangeSet,
) -> None:
    """Test if a set contains a ResidueRange."""
    range_a_in = ResidueRange("A", 12, 18)
    range_a_overlap = ResidueRange("A", 5, 15)  # Overlaps A:10-20 but not contained
    range_b_in = ResidueRange("B", 5, 15)
    range_b_partial = ResidueRange("B", 7, 12)
    range_a_out = ResidueRange("A", 1, 5)
    range_c = ResidueRange("C", 1, 10)

    assert range_a_in in residue_range_set_multi_chain
    assert (
        range_a_overlap not in residue_range_set_multi_chain
    )  # Containment requires full range
    assert range_b_in in residue_range_set_multi_chain
    assert range_b_partial in residue_range_set_multi_chain
    assert range_a_out not in residue_range_set_multi_chain
    assert range_c not in residue_range_set_multi_chain
    assert "A:12-18" not in residue_range_set_multi_chain  # Test wrong type


def test_empty_residue_range_set() -> None:
    """Test an empty ResidueRangeSet."""
    empty_set = ResidueRangeSet([])
    assert len(empty_set) == 0
    assert list(empty_set) == []
    assert str(empty_set) == ""
    assert ResidueCoordinate("A", 1) not in empty_set
    assert ResidueRange("A", 1, 5) not in empty_set


# --- Tests for ResidueRangeSet.is_adjacent_to ---


def test_residue_range_set_adjacent_simple() -> None:
    """Test simple adjacent ranges in the same chain."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    set2 = ResidueRangeSet.from_string("A:16-20")
    assert set1.is_adjacent_to(set2)
    assert set2.is_adjacent_to(set1)


def test_residue_range_set_adjacent_multiple_ranges() -> None:
    """Test adjacency when sets have multiple ranges, one pair adjacent."""
    set1 = ResidueRangeSet.from_string("A:10-15, B:1-5")
    set2 = ResidueRangeSet.from_string("C:1-10, A:16-20")
    set3 = ResidueRangeSet.from_string("B:6-10, D:1-5")
    assert set1.is_adjacent_to(set2)  # A:15 adjacent to A:16
    assert set2.is_adjacent_to(set1)
    assert set1.is_adjacent_to(set3)  # B:5 adjacent to B:6
    assert set3.is_adjacent_to(set1)
    assert not set2.is_adjacent_to(set3)  # No adjacent pairs


def test_residue_range_set_not_adjacent_gap() -> None:
    """Test non-adjacent ranges with a gap."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    set2 = ResidueRangeSet.from_string("A:17-20")  # Gap of 1 residue (16)
    assert not set1.is_adjacent_to(set2)
    assert not set2.is_adjacent_to(set1)


def test_residue_range_set_not_adjacent_overlap() -> None:
    """Test non-adjacent overlapping ranges."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    set2 = ResidueRangeSet.from_string("A:14-20")  # Overlap
    assert not set1.is_adjacent_to(set2)
    assert not set2.is_adjacent_to(set1)


def test_residue_range_set_not_adjacent_touching() -> None:
    """Test non-adjacent touching ranges (end == start)."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    set2 = ResidueRangeSet.from_string("A:15-20")  # Touching, not adjacent
    assert not set1.is_adjacent_to(set2)
    assert not set2.is_adjacent_to(set1)


def test_residue_range_set_not_adjacent_different_chains() -> None:
    """Test non-adjacent ranges in different chains."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    set2 = ResidueRangeSet.from_string("B:16-20")  # Different chains
    assert not set1.is_adjacent_to(set2)
    assert not set2.is_adjacent_to(set1)


def test_residue_range_set_adjacent_with_empty() -> None:
    """Test adjacency check with an empty set."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    empty_set = ResidueRangeSet([])
    assert not set1.is_adjacent_to(empty_set)
    assert not empty_set.is_adjacent_to(set1)
    assert not empty_set.is_adjacent_to(empty_set)


def test_residue_range_set_adjacent_type_error() -> None:
    """Test that comparing with a non-ResidueRangeSet raises TypeError."""
    set1 = ResidueRangeSet.from_string("A:10-15")
    with pytest.raises(TypeError):
        set1.is_adjacent_to("A:16-20")  # type: ignore


def test_residue_range_set_adjacent_to_range() -> None:
    """Test adjacency between a set and a single range."""
    set1 = ResidueRangeSet.from_string("A:10-15, B:1-5")
    range_adj_a = ResidueRange.from_string("A:16-20")
    range_adj_b = ResidueRange.from_string("B:6-10")
    range_non_adj = ResidueRange.from_string("C:1-10")
    range_gap_a = ResidueRange.from_string("A:17-20")

    assert set1.is_adjacent_to(range_adj_a)
    assert set1.is_adjacent_to(range_adj_b)
    assert not set1.is_adjacent_to(range_non_adj)
    assert not set1.is_adjacent_to(range_gap_a)


def test_residue_range_set_adjacent_to_coordinate() -> None:
    """Test adjacency between a set and a single coordinate."""
    set1 = ResidueRangeSet.from_string("A:10-15, B:1-5")
    coord_adj_a_before = ResidueCoordinate.from_string("A:9")
    coord_adj_a_after = ResidueCoordinate.from_string("A:16")
    coord_adj_b_before = ResidueCoordinate.from_string(
        "B:0"
    )  # Assuming 0 is valid index if it occurs
    coord_adj_b_after = ResidueCoordinate.from_string("B:6")
    coord_non_adj_gap = ResidueCoordinate.from_string("A:17")
    coord_non_adj_inside = ResidueCoordinate.from_string("B:3")
    coord_non_adj_chain = ResidueCoordinate.from_string("C:10")

    assert set1.is_adjacent_to(coord_adj_a_before)
    assert set1.is_adjacent_to(coord_adj_a_after)
    assert set1.is_adjacent_to(coord_adj_b_before)
    assert set1.is_adjacent_to(coord_adj_b_after)
    assert not set1.is_adjacent_to(coord_non_adj_gap)
    assert not set1.is_adjacent_to(coord_non_adj_inside)
    assert not set1.is_adjacent_to(coord_non_adj_chain)
