# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from unittest.mock import MagicMock

from flatprot.core import Structure, ResidueCoordinate, ResidueRangeSet
from flatprot.scene import Scene
from flatprot.scene.structure import (
    HelixSceneElement,
    CoilSceneElement,
    SheetSceneElement,
    CoilStyle,
)
from flatprot.scene.errors import CoordinateCalculationError

# --- Fixtures ---


@pytest.fixture
def mock_structure_data() -> np.ndarray:
    """Provides sample 3D coordinate data for testing."""
    # Simple linear structure: X=i, Y=0, Z=i*0.1
    return np.array([[i, 0, i * 0.1] for i in range(30)], dtype=float)


@pytest.fixture
def mock_structure(mock_structure_data: np.ndarray) -> MagicMock:
    """Creates a mock Structure object with predefined data and behavior."""
    structure = MagicMock(spec=Structure)
    structure.coordinates = mock_structure_data

    # Mock chain 'A' (residues 1-30)
    mock_chain_a = MagicMock(name="ChainA")
    chain_a_coord_map = {
        i: i - 1 for i in range(1, 31)
    }  # ResId (1-based) -> CoordIdx (0-based)

    def coordinate_index_a(res_idx):
        if res_idx in chain_a_coord_map:
            coord_idx = chain_a_coord_map[res_idx]
            if 0 <= coord_idx < len(structure.coordinates):
                return coord_idx
            else:
                raise IndexError(
                    f"Simulated: Coordinate index {coord_idx} out of bounds."
                )
        raise KeyError(f"Simulated: Residue {res_idx} not found in chain A map.")

    mock_chain_a.coordinate_index.side_effect = coordinate_index_a
    mock_chain_a.__contains__.side_effect = lambda res_idx: res_idx in chain_a_coord_map

    # Mock chain 'B' (empty)
    mock_chain_b = MagicMock(name="ChainB")
    mock_chain_b.coordinate_index.side_effect = KeyError(
        "Simulated: Residue not found in chain B map."
    )
    mock_chain_b.__contains__.return_value = False

    # Configure get_chain
    def get_chain_side_effect(chain_id):
        if chain_id == "A":
            return mock_chain_a
        elif chain_id == "B":
            return mock_chain_b
        else:
            raise KeyError(f"Simulated: Chain '{chain_id}' not found.")

    structure.__getitem__.side_effect = get_chain_side_effect
    return structure


@pytest.fixture
def scene_with_mock_structure(mock_structure: MagicMock) -> Scene:
    """Creates a Scene instance initialized with the mock Structure."""
    return Scene(structure=mock_structure)


# --- Test Cases ---


def test_helix_coordinate_generation(scene_with_mock_structure: Scene):
    """Test successful coordinate generation for a standard Helix."""
    scene = scene_with_mock_structure
    helix_range = ResidueRangeSet.from_string("A:5-15")  # 11 residues
    helix = HelixSceneElement(residue_range_set=helix_range)
    scene.add_node(helix)  # Add the *real* element

    coords = helix.get_coordinates(scene.structure)
    assert coords is not None
    assert isinstance(coords, np.ndarray)
    assert coords.shape[1] == 3  # X, Y, Depth
    # Check Z coords at start/end (indices 4 and 14 in mock data)
    expected_start_z = scene.structure.coordinates[4][2]
    expected_end_z = scene.structure.coordinates[14][2]
    num_wave_points = len(coords) // 2
    assert np.isclose(coords[0][2], expected_start_z)
    assert np.isclose(coords[-1][2], expected_start_z)  # First bottom point
    assert np.isclose(coords[num_wave_points - 1][2], expected_end_z)  # Last top point
    assert np.isclose(coords[num_wave_points][2], expected_end_z)  # Last bottom point


def test_coil_coordinate_generation(scene_with_mock_structure: Scene):
    """Test successful coordinate generation for a standard Coil."""
    scene = scene_with_mock_structure
    coil_range = ResidueRangeSet.from_string("A:1-11")  # 11 residues
    # Use specific style for predictable smoothing
    coil_style = CoilStyle(smoothing_factor=0.5)
    coil = CoilSceneElement(residue_range_set=coil_range, style=coil_style)
    scene.add_node(coil)

    coords = coil.get_coordinates(scene.structure)
    assert coords is not None
    assert isinstance(coords, np.ndarray)
    assert coords.shape[1] == 3  # X, Y, Depth
    # Expect max(2, int(11 * 0.5)) = 5 points
    assert len(coords) == 5
    # Check start/end points match original data
    np.testing.assert_array_almost_equal(coords[0], scene.structure.coordinates[0])
    np.testing.assert_array_almost_equal(coords[-1], scene.structure.coordinates[10])


def test_helix_short_helix_as_line(scene_with_mock_structure: Scene):
    """Test a helix shorter than min_helix_length renders as a line."""
    scene = scene_with_mock_structure
    # Default min_helix_length is 4
    helix_range = ResidueRangeSet.from_string("A:1-3")  # 3 residues < 4
    helix = HelixSceneElement(residue_range_set=helix_range)
    scene.add_node(helix)

    coords = helix.get_coordinates(scene.structure)
    assert coords is not None
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (2, 3)  # Should only have start and end points
    np.testing.assert_array_almost_equal(coords[0], scene.structure.coordinates[0])
    np.testing.assert_array_almost_equal(coords[1], scene.structure.coordinates[2])


def test_sheet_coordinate_generation(scene_with_mock_structure: Scene):
    """Test successful coordinate generation for a standard Sheet."""
    scene = scene_with_mock_structure
    sheet_range = ResidueRangeSet.from_string("A:1-11")  # 11 residues
    sheet = SheetSceneElement(residue_range_set=sheet_range)
    scene.add_node(sheet)

    coords = sheet.get_coordinates(scene.structure)
    assert coords is not None
    assert isinstance(coords, np.ndarray)
    # Expect 3 points for the arrow: [left_base, right_base, tip]
    assert coords.shape == (3, 3)

    # 1. Verify Tip Point (coords[2]) matches end residue (index 10)
    end_coord = scene.structure.coordinates[10]
    np.testing.assert_array_almost_equal(coords[2], end_coord)

    # 2. Verify Base Midpoint aligns with start residue (index 0) in XY
    start_coord = scene.structure.coordinates[0]
    base_midpoint = (coords[0] + coords[1]) / 2.0
    np.testing.assert_array_almost_equal(base_midpoint[:2], start_coord[:2])
    # Check base depth (should be average of start/end Z)
    avg_depth = (start_coord[2] + end_coord[2]) / 2.0
    assert np.isclose(base_midpoint[2], avg_depth)

    # 3. Verify Base Vector is perpendicular to the main direction vector (in XY plane)
    base_vector_xy = coords[1][:2] - coords[0][:2]
    direction_vector_xy = end_coord[:2] - start_coord[:2]
    # Ensure direction_vector is not zero length before dot product
    assert np.linalg.norm(direction_vector_xy) > 1e-6
    dot_product = np.dot(base_vector_xy, direction_vector_xy)
    assert np.isclose(
        dot_product, 0.0
    ), f"Dot product should be close to 0, but was {dot_product}"

    # 4. Verify Base Width (optional but good)
    expected_base_width = sheet.style.stroke_width * sheet.style.arrow_width_factor
    actual_base_width = np.linalg.norm(base_vector_xy)
    assert np.isclose(actual_base_width, expected_base_width)


# --- Coordinate Calculation Error Tests ---


@pytest.mark.parametrize(
    "ElementClass", [HelixSceneElement, CoilSceneElement, SheetSceneElement]
)
def test_coord_error_on_missing_residue(
    mock_structure: MagicMock, scene_with_mock_structure: Scene, ElementClass: type
):
    """Test Element raises CoordinateCalculationError if structure lookup fails (KeyError)."""
    scene = scene_with_mock_structure
    # Configure chain 'A' to fail for residue 10
    original_coord_index = mock_structure["A"].coordinate_index.side_effect

    def failing_coord_index(res_idx):
        if res_idx == 10:
            raise KeyError("Simulated residue 10 missing")
        # Need to handle the case where original_coord_index is not callable if it was overwritten before
        if callable(original_coord_index):
            return original_coord_index(res_idx)
        else:
            # Fallback or raise different error if needed
            raise TypeError("Original side effect is not callable")

    mock_structure["A"].coordinate_index.side_effect = failing_coord_index

    element_range = ResidueRangeSet.from_string("A:5-15")  # Range includes residue 10
    element = ElementClass(residue_range_set=element_range)
    scene.add_node(element)

    with pytest.raises(
        CoordinateCalculationError,
        match="Error getting original coordinates|Error fetching coordinates",
    ):
        element.get_coordinates(scene.structure)


@pytest.mark.parametrize(
    "ElementClass", [HelixSceneElement, CoilSceneElement, SheetSceneElement]
)
def test_coord_error_on_invalid_coord_index(
    mock_structure: MagicMock, scene_with_mock_structure: Scene, ElementClass: type
):
    """Test Element raises CoordinateCalculationError if coord index is out of bounds."""
    scene = scene_with_mock_structure
    # Configure chain 'A' to return an invalid index (e.g., 100) for residue 10
    original_coord_index = mock_structure["A"].coordinate_index.side_effect

    def invalid_coord_index(res_idx):
        if res_idx == 10:
            return 100  # Invalid index for mock_structure_data (size 30)
        if callable(original_coord_index):
            return original_coord_index(res_idx)
        else:
            raise TypeError("Original side effect is not callable")

    mock_structure["A"].coordinate_index.side_effect = invalid_coord_index

    element_range = ResidueRangeSet.from_string("A:5-15")  # Range includes residue 10
    element = ElementClass(residue_range_set=element_range)
    scene.add_node(element)

    with pytest.raises(CoordinateCalculationError, match="index .* out of bounds"):
        element.get_coordinates(scene.structure)


@pytest.mark.parametrize(
    "ElementClass", [HelixSceneElement, CoilSceneElement, SheetSceneElement]
)
def test_coord_error_on_missing_chain(
    mock_structure: MagicMock, scene_with_mock_structure: Scene, ElementClass: type
):
    """Test Element raises CoordinateCalculationError if the chain is not found."""
    scene = scene_with_mock_structure
    element_range = ResidueRangeSet.from_string("Z:5-15")  # Chain Z does not exist
    element = ElementClass(residue_range_set=element_range)
    scene.add_node(element)

    with pytest.raises(
        CoordinateCalculationError,
        match="Error fetching coordinates|Error getting original coordinates",
    ):
        element.get_coordinates(scene.structure)


def test_helix_error_zero_length_endpoints(scene_with_mock_structure: Scene):
    """Test Helix raises CoordinateCalcError if start/end points are identical."""
    scene = scene_with_mock_structure
    structure = scene.structure
    # Make coordinates for residues 5 and 15 identical (indices 4 and 14)
    structure.coordinates[14] = structure.coordinates[4].copy()

    helix_range = ResidueRangeSet.from_string("A:5-15")
    helix = HelixSceneElement(residue_range_set=helix_range)
    scene.add_node(helix)

    # get_coordinates itself should raise the error when calculate_zigzag_points returns None
    with pytest.raises(
        CoordinateCalculationError,
        match="Could not generate zigzag points.*zero length",
    ):
        helix.get_coordinates(structure)


# --- get_coordinate_at_residue Tests ---


def test_helix_get_coord_at_residue(scene_with_mock_structure: Scene):
    """Test retrieving coordinate for a specific residue on a helix."""
    scene = scene_with_mock_structure
    structure = scene.structure
    helix_range = ResidueRangeSet.from_string("A:5-15")  # 11 residues (indices 4-14)
    helix = HelixSceneElement(residue_range_set=helix_range)
    scene.add_node(helix)
    # It's important to call get_coordinates first to populate internal cache if get_coordinate_at_residue relies on it
    helix.get_coordinates(structure)

    # Test middle residue (A:10), corresponds to index 9 in structure.coordinates
    target_residue_mid = ResidueCoordinate("A", 10)
    coord_mid = helix.get_coordinate_at_residue(target_residue_mid, structure)
    assert coord_mid is not None
    assert coord_mid.shape == (3,)

    # Calculate the expected coordinate using the same logic as the implementation:
    # interpolate between the midpoint of the top/bottom edges.
    display_coords = helix._cached_display_coords  # Get the calculated ribbon points
    assert display_coords is not None  # Should have been calculated earlier
    num_wave_points = len(display_coords) // 2
    orig_len = helix._original_coords_len
    assert orig_len is not None
    original_sequence_index = (
        target_residue_mid.residue_index - helix_range.ranges[0].start
    )  # Index is 5 (10-5)
    mapped_wave_frac = (original_sequence_index * (num_wave_points - 1)) / (
        orig_len - 1
    )
    idx_low = int(np.floor(mapped_wave_frac))
    idx_high = min(idx_low + 1, num_wave_points - 1)
    idx_low = min(idx_low, num_wave_points - 1)
    frac = mapped_wave_frac - idx_low
    top_low = display_coords[idx_low]
    top_high = display_coords[idx_high]
    bottom_low = display_coords[len(display_coords) - 1 - idx_low]
    bottom_high = display_coords[len(display_coords) - 1 - idx_high]
    interp_top = top_low * (1 - frac) + top_high * frac
    interp_bottom = bottom_low * (1 - frac) + bottom_high * frac
    expected_mid = (interp_top + interp_bottom) / 2.0

    # Now assert against the correctly calculated expected coordinate
    np.testing.assert_array_almost_equal(coord_mid, expected_mid, decimal=5)

    # Test start residue (A:5), corresponds to index 4
    target_residue_start = ResidueCoordinate("A", 5)
    coord_start = helix.get_coordinate_at_residue(target_residue_start, structure)
    assert coord_start is not None
    # Expected start should be close to the first coordinate used (index 4)
    np.testing.assert_array_almost_equal(
        coord_start, structure.coordinates[4], decimal=5
    )

    # Test end residue (A:15), corresponds to index 14
    target_residue_end = ResidueCoordinate("A", 15)
    coord_end = helix.get_coordinate_at_residue(target_residue_end, structure)
    assert coord_end is not None
    # Expected end should be close to the last coordinate used (index 14)
    np.testing.assert_array_almost_equal(
        coord_end, structure.coordinates[14], decimal=5
    )

    # Test residue outside range
    outside_residue = ResidueCoordinate("A", 20)
    assert helix.get_coordinate_at_residue(outside_residue, structure) is None
    wrong_chain_residue = ResidueCoordinate("B", 10)
    assert helix.get_coordinate_at_residue(wrong_chain_residue, structure) is None


def test_sheet_get_coord_at_residue(scene_with_mock_structure: Scene):
    """Test retrieving coordinate for a specific residue on a sheet line."""
    scene = scene_with_mock_structure
    structure = scene.structure
    sheet_range = ResidueRangeSet.from_string("A:1-11")  # 11 residues (indices 0-10)
    sheet = SheetSceneElement(residue_range_set=sheet_range)
    scene.add_node(sheet)
    # Call get_coordinates if needed by implementation (assuming simple line doesn't cache like coil)
    sheet.get_coordinates(structure)

    # Test middle residue (A:6), original index 5
    target_residue_mid = ResidueCoordinate("A", 6)
    coord_mid = sheet.get_coordinate_at_residue(target_residue_mid, structure)
    assert coord_mid is not None
    assert coord_mid.shape == (3,)

    # Calculate expected midpoint based on implementation logic:
    # Interpolation between base midpoint and tip point
    display_coords = sheet._cached_display_coords  # Need the calculated arrow points
    assert display_coords is not None and len(display_coords) == 3
    base_midpoint = (display_coords[0] + display_coords[1]) / 2.0
    tip_point = display_coords[2]
    orig_len = sheet._original_coords_len
    assert orig_len is not None and orig_len > 1
    original_sequence_index = (
        target_residue_mid.residue_index - sheet_range.ranges[0].start
    )  # Index 5
    frac = original_sequence_index / (orig_len - 1)  # Frac = 5 / (11 - 1) = 0.5
    expected_mid = base_midpoint * (1 - frac) + tip_point * frac

    # Assert against the correctly calculated expected coordinate
    np.testing.assert_array_almost_equal(coord_mid, expected_mid)

    # Test start residue (A:1), original index 0
    target_residue_start = ResidueCoordinate("A", 1)
    coord_start = sheet.get_coordinate_at_residue(target_residue_start, structure)
    assert coord_start is not None
    # For frac=0, the result should be the base_midpoint
    np.testing.assert_array_almost_equal(coord_start, base_midpoint)

    # Test end residue (A:11), original index 10
    target_residue_end = ResidueCoordinate("A", 11)
    coord_end = sheet.get_coordinate_at_residue(target_residue_end, structure)
    assert coord_end is not None
    # For frac=1, the result should be the tip_point
    np.testing.assert_array_almost_equal(coord_end, tip_point)

    # Test residue outside range
    outside_residue = ResidueCoordinate("A", 20)
    assert sheet.get_coordinate_at_residue(outside_residue, structure) is None
    wrong_chain_residue = ResidueCoordinate("B", 6)
    assert sheet.get_coordinate_at_residue(wrong_chain_residue, structure) is None


def test_coil_get_coord_at_residue(scene_with_mock_structure: Scene):
    """Test retrieving coordinate for a specific residue on a smoothed coil."""
    scene = scene_with_mock_structure
    structure = scene.structure
    coil_range = ResidueRangeSet.from_string("A:1-11")  # 11 residues (indices 0-10)
    # Smoothing factor 0.2 keeps 2 points: indices 0 and 10
    coil_style = CoilStyle(smoothing_factor=0.2)
    coil = CoilSceneElement(residue_range_set=coil_range, style=coil_style)
    scene.add_node(coil)
    # Call get_coordinates to populate cache
    smoothed_coords = coil.get_coordinates(structure)
    assert len(smoothed_coords) == 2  # Verify smoothing happened as expected

    # Test middle residue (A:6), corresponds to original index 5
    target_residue_mid = ResidueCoordinate("A", 6)
    coord_mid = coil.get_coordinate_at_residue(target_residue_mid, structure)
    assert coord_mid is not None
    assert coord_mid.shape == (3,)
    # Expected is interpolation between start(idx 0) and end(idx 10)
    # Original index 5 is halfway through the 11 residues (0-10)
    # Frac = 5 / (11 - 1) = 0.5
    expected_mid = structure.coordinates[0] * 0.5 + structure.coordinates[10] * 0.5
    np.testing.assert_array_almost_equal(coord_mid, expected_mid)

    # Test start residue (A:1), original index 0
    target_residue_start = ResidueCoordinate("A", 1)
    coord_start = coil.get_coordinate_at_residue(target_residue_start, structure)
    assert coord_start is not None
    np.testing.assert_array_almost_equal(coord_start, structure.coordinates[0])

    # Test end residue (A:11), original index 10
    target_residue_end = ResidueCoordinate("A", 11)
    coord_end = coil.get_coordinate_at_residue(target_residue_end, structure)
    assert coord_end is not None
    np.testing.assert_array_almost_equal(coord_end, structure.coordinates[10])

    # Test residue outside range
    outside_residue = ResidueCoordinate("A", 20)
    assert coil.get_coordinate_at_residue(outside_residue, structure) is None
    wrong_chain_residue = ResidueCoordinate("B", 6)
    assert coil.get_coordinate_at_residue(wrong_chain_residue, structure) is None
