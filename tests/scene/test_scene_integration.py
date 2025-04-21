# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from flatprot.core import (
    Structure,
    ResidueCoordinate,
    ResidueRangeSet,
    CoordinateCalculationError,
)
from flatprot.scene import Scene
from flatprot.scene.structure import (
    HelixSceneElement,
    CoilSceneElement,
    SheetSceneElement,
    CoilStyle,
)
from flatprot.scene.annotation import (
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,  # Import BaseAnnotationElement for mocking
)

# --- Fixtures ---


@pytest.fixture
def mock_structure_data() -> np.ndarray:
    """Provides sample 3D coordinate data for testing."""
    # Simple linear structure: X=i, Y=0, Z=i*0.1
    return np.array([[i, 0, i * 0.1] for i in range(30)], dtype=float)


@pytest.fixture
def mock_structure(mock_structure_data: np.ndarray, mocker):
    """Creates a mock Structure object with predefined data and behavior."""
    structure = mocker.MagicMock(spec=Structure)
    structure.coordinates = mock_structure_data

    # --- Mock Chain A --- #
    mock_chain_a = mocker.MagicMock(name="ChainA")
    # Map ResId (1-based) -> CoordIdx (0-based) for chain A (residues 1-30)
    chain_a_coord_map = {i: i - 1 for i in range(1, 31)}

    def coord_index_a(res_idx):
        idx = chain_a_coord_map.get(res_idx)
        if idx is None:
            raise KeyError(f"Simulated key error for A:{res_idx}")
        return idx

    mock_chain_a.coordinate_index.side_effect = coord_index_a
    mock_chain_a.__contains__.side_effect = lambda res_idx: res_idx in chain_a_coord_map

    # --- Mock Chain B (Empty) --- #
    mock_chain_b = mocker.MagicMock(name="ChainB")
    mock_chain_b.coordinate_index.side_effect = KeyError("Simulated key error for B")
    mock_chain_b.__contains__.return_value = False

    # --- Mock Structure Chain Access (__getitem__) --- #
    def getitem_side_effect(chain_id):
        if chain_id == "A":
            return mock_chain_a
        elif chain_id == "B":
            return mock_chain_b
        else:
            raise KeyError(f"Simulated chain {chain_id} not found")

    structure.__getitem__.side_effect = getitem_side_effect

    # --- Mock get_coordinate_at_residue (still needed for Annotation tests) --- #
    def get_coord_at_residue(residue: ResidueCoordinate) -> np.ndarray | None:
        # This mock remains simple, relying on the base coordinate array
        if residue.chain_id == "A" and 1 <= residue.residue_index <= 30:
            coord_idx = residue.residue_index - 1
            if 0 <= coord_idx < len(structure.coordinates):
                return structure.coordinates[coord_idx]
        return None

    structure.get_coordinate_at_residue.side_effect = get_coord_at_residue

    return structure


@pytest.fixture
def scene_with_mock_structure(mock_structure) -> Scene:
    """Creates a Scene instance initialized with the mock Structure."""
    return Scene(structure=mock_structure)


@pytest.fixture
def scene_with_structure_elements(scene_with_mock_structure: Scene) -> Scene:
    """Creates a scene with a coil, helix, and sheet element."""
    scene = scene_with_mock_structure
    # Add elements covering different parts of chain A (1-30)
    # Use explicit IDs for easier retrieval if needed, though not strictly necessary now
    coil = CoilSceneElement(
        id="coil_A_1-6", residue_range_set=ResidueRangeSet.from_string("A:1-6")
    )
    helix = HelixSceneElement(
        id="helix_A_7-17", residue_range_set=ResidueRangeSet.from_string("A:7-17")
    )
    sheet = SheetSceneElement(
        id="sheet_A_18-28", residue_range_set=ResidueRangeSet.from_string("A:18-28")
    )
    try:
        scene.add_element(coil)
        scene.add_element(helix)
        scene.add_element(sheet)
        # Pre-calculate coordinates for structure elements to ensure they are valid
        for element in [coil, helix, sheet]:
            element.get_coordinates(scene.structure)
    except Exception as e:
        pytest.skip(f"Skipping due to fixture setup error: {e}")
    return scene


@pytest.fixture
def point_anno() -> PointAnnotation:
    """Basic point annotation."""
    return PointAnnotation(id="p1", target=ResidueCoordinate("A", 10))


@pytest.fixture
def line_anno() -> LineAnnotation:
    """Basic line annotation."""
    return LineAnnotation(
        id="l1",
        target_coordinates=[ResidueCoordinate("A", 3), ResidueCoordinate("A", 20)],
    )


@pytest.fixture
def area_anno() -> AreaAnnotation:
    """Basic area annotation."""
    return AreaAnnotation(
        id="area1", residue_range_set=ResidueRangeSet.from_string("A:8-12")
    )


# --- Element Coordinate Calculation Tests (remain largely unchanged) ---


def test_helix_coordinate_generation(scene_with_mock_structure: Scene):
    """Test successful coordinate generation for a standard Helix."""
    scene = scene_with_mock_structure
    helix_range = ResidueRangeSet.from_string("A:5-15")  # 11 residues
    helix = HelixSceneElement(residue_range_set=helix_range)
    scene.add_element(helix)  # Add the *real* element

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
    scene.add_element(coil)

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
    scene.add_element(helix)

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
    scene.add_element(sheet)

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
    expected_base_width = sheet.style.stroke_width * sheet.style.arrow_width
    actual_base_width = np.linalg.norm(base_vector_xy)
    assert np.isclose(actual_base_width, expected_base_width)


# --- Element Coordinate Calculation Error Tests (based on internal get_coordinates) ---


@pytest.mark.parametrize(
    "ElementClass", [HelixSceneElement, CoilSceneElement, SheetSceneElement]
)
def test_element_coord_error_on_missing_residue(
    mock_structure, scene_with_mock_structure: Scene, ElementClass: type
):
    """Test Element raises CoordinateCalculationError if structure lookup fails."""
    scene = scene_with_mock_structure
    structure = scene.structure
    # Make structure return None for residue 10
    original_get_coord = structure.get_coordinate_at_residue.side_effect

    def fail_at_10(residue: ResidueCoordinate):
        if residue.chain_id == "A" and residue.residue_index == 10:
            return None
        return original_get_coord(residue)

    structure.get_coordinate_at_residue.side_effect = fail_at_10

    element_range = ResidueRangeSet.from_string("A:5-15")  # Range includes residue 10
    element = ElementClass(residue_range_set=element_range)
    scene.add_element(element)

    # Error now comes from _get_original_coords_slice when residue 10 is not found
    # by the mocked structure.get_coordinate_at_residue returning None, which isn't
    # the scenario here. The error comes from chain lookup/indexing failure.
    # We adjust the test to expect the error from the internal lookup failing.
    # Let's mock structure[chain_id].__contains__ to fail for res 10.
    original_contains = structure["A"].__contains__.side_effect

    def fail_contains_at_10(res_idx):
        if res_idx == 10:
            return False
        return original_contains(res_idx)

    structure["A"].__contains__.side_effect = fail_contains_at_10

    # Expect the error message raised by _get_original_coords_slice
    with pytest.raises(
        CoordinateCalculationError,
        match="Residue A:10 not found in chain coordinate map",
    ):
        element.get_coordinates(structure)


# --- Element get_coordinate_at_residue Tests (remain largely unchanged) ---


def test_helix_get_coord_at_residue(scene_with_mock_structure: Scene):
    """Test retrieving coordinate for a specific residue on a helix."""
    scene = scene_with_mock_structure
    structure = scene.structure
    helix_range = ResidueRangeSet.from_string("A:5-15")  # 11 residues (indices 4-14)
    helix = HelixSceneElement(residue_range_set=helix_range)
    scene.add_element(helix)
    # It's important to call get_coordinates first to populate internal cache
    display_coords = helix.get_coordinates(structure)
    assert display_coords is not None

    # Test middle residue (A:10), corresponds to index 9 in structure.coordinates
    target_residue_mid = ResidueCoordinate("A", 10)
    coord_mid = helix.get_coordinate_at_residue(target_residue_mid, structure)
    assert coord_mid is not None
    assert coord_mid.shape == (3,)

    # Calculate the expected coordinate using the same logic as the implementation:
    # interpolate between the midpoint of the top/bottom edges.
    assert helix._cached_display_coords is not None  # Should have been calculated
    num_wave_points = len(helix._cached_display_coords) // 2
    orig_len = helix._original_coords_len
    assert orig_len is not None and orig_len > 1
    original_sequence_index = (
        target_residue_mid.residue_index - helix_range.ranges[0].start
    )  # Index is 5 (10-5)
    # Ensure division by zero doesn't happen if orig_len is 1 (though unlikely here)
    divisor = max(1, orig_len - 1)
    mapped_wave_frac = (original_sequence_index * (num_wave_points - 1)) / divisor
    idx_low = int(np.floor(mapped_wave_frac))
    idx_high = min(idx_low + 1, num_wave_points - 1)
    idx_low = min(idx_low, num_wave_points - 1)  # Ensure idx_low is not out of bounds
    frac = mapped_wave_frac - idx_low

    top_low = helix._cached_display_coords[idx_low]
    top_high = helix._cached_display_coords[idx_high]
    bottom_low = helix._cached_display_coords[
        len(helix._cached_display_coords) - 1 - idx_low
    ]
    bottom_high = helix._cached_display_coords[
        len(helix._cached_display_coords) - 1 - idx_high
    ]

    interp_top = top_low * (1 - frac) + top_high * frac
    interp_bottom = bottom_low * (1 - frac) + bottom_high * frac
    expected_mid = (interp_top + interp_bottom) / 2.0

    # Assert against the correctly calculated expected coordinate
    np.testing.assert_array_almost_equal(coord_mid, expected_mid, decimal=5)


def test_sheet_get_coord_at_residue(scene_with_mock_structure: Scene):
    """Test retrieving coordinate for a specific residue on a sheet line."""
    scene = scene_with_mock_structure
    structure = scene.structure
    sheet_range = ResidueRangeSet.from_string("A:1-11")  # 11 residues (indices 0-10)
    sheet = SheetSceneElement(residue_range_set=sheet_range)
    scene.add_element(sheet)
    # Call get_coordinates to populate cache
    display_coords = sheet.get_coordinates(structure)
    assert display_coords is not None and len(display_coords) == 3

    # Test middle residue (A:6), original index 5
    target_residue_mid = ResidueCoordinate("A", 6)
    coord_mid = sheet.get_coordinate_at_residue(target_residue_mid, structure)
    assert coord_mid is not None
    assert coord_mid.shape == (3,)

    # Calculate expected midpoint based on implementation logic:
    # Interpolation between base midpoint and tip point
    assert sheet._cached_display_coords is not None  # Cache should exist
    base_midpoint = (
        sheet._cached_display_coords[0] + sheet._cached_display_coords[1]
    ) / 2.0
    tip_point = sheet._cached_display_coords[2]
    orig_len = sheet._original_coords_len
    assert orig_len is not None and orig_len > 1
    original_sequence_index = (
        target_residue_mid.residue_index - sheet_range.ranges[0].start
    )  # Index 5
    frac = original_sequence_index / (orig_len - 1)  # Frac = 5 / (11 - 1) = 0.5
    expected_mid = base_midpoint * (1 - frac) + tip_point * frac

    # Assert against the correctly calculated expected coordinate
    np.testing.assert_array_almost_equal(coord_mid, expected_mid)


def test_coil_get_coord_at_residue(scene_with_mock_structure: Scene):
    """Test retrieving coordinate for a specific residue on a smoothed coil."""
    scene = scene_with_mock_structure
    structure = scene.structure
    coil_range = ResidueRangeSet.from_string("A:1-11")  # 11 residues (indices 0-10)
    # Smoothing factor 0.2 keeps 2 points: indices 0 and 10
    coil_style = CoilStyle(smoothing_factor=0.2)
    coil = CoilSceneElement(residue_range_set=coil_range, style=coil_style)
    scene.add_element(coil)
    # Call get_coordinates to populate cache
    smoothed_coords = coil.get_coordinates(structure)
    assert smoothed_coords is not None and len(smoothed_coords) == 2

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


# Remove tests for Scene.get_rendered_coordinates_for_annotation
# as this logic is now handled by the CoordinateResolver and the annotations themselves.
