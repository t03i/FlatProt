# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from xml.etree import ElementTree as ET
from typing import Optional

import pytest
import numpy as np
from pydantic_extra_types.color import Color

from flatprot.core import (
    Structure,
    ResidueRangeSet,
    ResidueCoordinate,
)
from flatprot.scene import (
    Scene,
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
    HelixSceneElement,
)
from flatprot.renderers import SVGRenderer


# --- Helper Functions ---


def _parse_svg(svg_output: str) -> ET.Element:
    """Helper to parse SVG and fail test on error."""
    try:
        # Ensure output is not empty before parsing
        if not svg_output or not svg_output.strip():
            pytest.fail("Generated SVG output is empty or whitespace only.")
        return ET.fromstring(svg_output)
    except ET.ParseError as e:
        pytest.fail(
            f"Generated SVG is not well-formed XML: {e}\\nOutput:\\n{svg_output}"
        )


def _parse_svg_path_d(d_str: str) -> Optional[np.ndarray]:
    """Parses a simple SVG path 'd' string (M L L... Z) into coordinates."""
    if not d_str:
        return None
    # Clean the string: remove commands, commas, ensure space separation
    cleaned_str = (
        d_str.upper()
        .replace(",", " ")
        .replace("M", "")
        .replace("L", "")
        .replace("Z", "")
        .strip()
    )
    parts = cleaned_str.split()

    try:
        # Convert all parts to float
        coords_flat = [float(part) for part in parts]

        # Check if we have an even number of coordinates for pairing
        if len(coords_flat) % 2 != 0:
            print(f"Warning: Odd number of coordinates after parsing path: {d_str}")
            return None

        num_points = len(coords_flat) // 2
        if num_points == 0:
            return np.empty((0, 2))

        # Reshape into N x 2 array
        return np.array(coords_flat).reshape((num_points, 2))

    except ValueError as e:
        print(
            f"Warning: Could not parse float in path coordinates: {d_str} | Error: {e}"
        )
        return None
    except Exception as e:  # Catch other potential errors during parsing/reshaping
        print(f"Warning: Unexpected error parsing path: {d_str} | Error: {e}")
        return None


# --- Fixtures ---


@pytest.fixture
def mock_structure_coords() -> np.ndarray:
    """Provides coordinates for the mock structure."""
    coords = np.zeros((10, 3))
    coords[:, 0] = np.arange(10) * 5  # X coordinates: 0, 5, 10, ..., 45
    coords[:, 1] = 10  # Constant Y
    # Make points 4, 5, 6 slightly non-collinear to avoid 2-point area case
    coords[5, 1] = 10.1
    return coords


@pytest.fixture
def mock_structure(mock_structure_coords: np.ndarray, mocker) -> Structure:
    """Provides a mock Structure object with mocked chain access and coordinate lookup."""
    structure = mocker.MagicMock(spec=Structure)
    structure.id = "mock_struct_anno"
    structure.coordinates = mock_structure_coords

    # --- Mock Chain A --- #
    mock_chain_a = mocker.MagicMock(name="ChainA")
    # Map ResId (1-based) -> CoordIdx (0-based) for chain A (residues 1-10)
    chain_a_coord_map = {i: i - 1 for i in range(1, 11)}

    def coord_index_a(res_idx):
        idx = chain_a_coord_map.get(res_idx)
        if idx is None:
            raise KeyError(f"Simulated key error for A:{res_idx}")
        return idx

    mock_chain_a.coordinate_index.side_effect = coord_index_a
    mock_chain_a.__contains__.side_effect = lambda res_idx: res_idx in chain_a_coord_map
    mock_chain_a.id = "A"  # Ensure the mock chain has an ID

    # --- Mock Structure Chain Access (__getitem__) --- #
    def getitem_side_effect(chain_id):
        if chain_id == "A":
            return mock_chain_a
        # Add more mock chains if needed for other tests
        else:
            raise KeyError(f"Simulated chain {chain_id} not found")

    structure.__getitem__.side_effect = getitem_side_effect
    # Also mock the .chains attribute if it's accessed directly elsewhere
    structure.chains = {"A": mock_chain_a}

    # --- Mock get_coordinate_at_residue (still needed for Annotation tests) --- #
    def get_coord_at_res(residue: ResidueCoordinate) -> Optional[np.ndarray]:
        if residue.chain_id == "A" and 1 <= residue.residue_index <= 10:
            coord_idx = residue.residue_index - 1
            if 0 <= coord_idx < len(mock_structure_coords):
                return mock_structure_coords[coord_idx]
        return None

    structure.get_coordinate_at_residue = mocker.MagicMock(side_effect=get_coord_at_res)

    return structure


@pytest.fixture
def empty_scene(mock_structure: Structure) -> Scene:
    """Provides an empty Scene object based on the mock structure."""
    # No change needed here, just uses the mock_structure
    return Scene(structure=mock_structure)


# --- Structure Element Fixture (for annotations to target) ---


@pytest.fixture
def helix_A_3_7(
    mock_structure: Structure,
) -> HelixSceneElement:
    """Provides a HelixSceneElement targeting Chain A, residues 3-7."""
    # No mocking needed inside this fixture anymore
    helix = HelixSceneElement(residue_range_set=ResidueRangeSet.from_string("A:3-7"))
    # Ensure the element can calculate its own coords if needed internally
    try:
        _ = helix.get_coordinates(mock_structure)
    except Exception as e:
        print(f"Warning: helix_A_3_7 fixture failed internal get_coordinates: {e}")
        # Allow test to proceed, maybe element coords aren't needed by annotation render test
        pass
    return helix


# --- Annotation Fixtures (targeting residues) ---


@pytest.fixture
def point_anno_A5() -> PointAnnotation:
    """Provides a PointAnnotation targeting residue A:5."""
    return PointAnnotation(id="point_on_A5", target=ResidueCoordinate("A", 5))


@pytest.fixture
def line_anno_A4_A6() -> LineAnnotation:
    """Provides a LineAnnotation targeting residues A:4 and A:6."""
    return LineAnnotation(
        id="line_A4_A6",
        start_coordinate=ResidueCoordinate("A", 4),
        end_coordinate=ResidueCoordinate("A", 6),
    )


@pytest.fixture
def area_anno_A5_A7() -> AreaAnnotation:
    """Provides an AreaAnnotation targeting residue range A:5-7."""
    return AreaAnnotation(
        id="area_A5_A7", residue_range_set=ResidueRangeSet.from_string("A:5-7")
    )


# --- Scene Fixtures with Structure + Annotations ---


@pytest.fixture
def scene_with_helix(empty_scene: Scene, helix_A_3_7: HelixSceneElement) -> Scene:
    """Scene containing only the helix A:3-7."""
    empty_scene.add_element(helix_A_3_7)
    return empty_scene


@pytest.fixture
def scene_with_helix_and_point(
    scene_with_helix: Scene, point_anno_A5: PointAnnotation
) -> Scene:
    """Scene with helix A:3-7 and a point annotation on A:5."""
    scene_with_helix.add_element(point_anno_A5)
    return scene_with_helix


@pytest.fixture
def scene_with_helix_and_line(
    scene_with_helix: Scene, line_anno_A4_A6: LineAnnotation
) -> Scene:
    """Scene with helix A:3-7 and a line annotation from A:4 to A:6."""
    scene_with_helix.add_element(line_anno_A4_A6)
    return scene_with_helix


@pytest.fixture
def scene_with_helix_and_area(
    scene_with_helix: Scene, area_anno_A5_A7: AreaAnnotation
) -> Scene:
    """Scene with helix A:3-7 and an area annotation covering A:5-7."""
    scene_with_helix.add_element(area_anno_A5_A7)
    return scene_with_helix


# --- Tests ---


def test_render_point_annotation(
    scene_with_helix_and_point: Scene,
    point_anno_A5: PointAnnotation,
    mock_structure_coords: np.ndarray,
) -> None:
    """
    Tests if a PointAnnotation renders a circle at the expected coordinates.
    """
    scene = scene_with_helix_and_point
    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # --- Get the ACTUAL base coordinate from the element via resolver --- #
    # This represents the point where the annotation should anchor
    # We use the resolver directly here to simulate what the renderer does internally
    resolver = scene.resolver
    base_coords = resolver.resolve(point_anno_A5.target_coordinate)
    assert base_coords is not None

    # --- Calculate expected SVG position based on resolved coord and style --- #
    expected_cx = base_coords[0] + point_anno_A5.style.offset[0]
    expected_cy = base_coords[1] + point_anno_A5.style.offset[1]

    # Find the corresponding circle marker
    marker_id = f"{point_anno_A5.id}-marker"
    circles = root.findall(
        f".//svg:circle[@class='annotation point-marker'][@id='{marker_id}']",
        namespaces=ns,
    )
    assert len(circles) == 1, f"Expected 1 point marker circle, found {len(circles)}"

    circle = circles[0]
    actual_cx = float(circle.attrib.get("cx", "NaN"))
    actual_cy = float(circle.attrib.get("cy", "NaN"))
    actual_r = float(circle.attrib.get("r", "NaN"))

    assert actual_cx == pytest.approx(expected_cx)
    assert actual_cy == pytest.approx(expected_cy)
    assert actual_r == pytest.approx(point_anno_A5.style.marker_radius)
    assert circle.attrib.get("fill") == point_anno_A5.style.color.as_hex()


def test_render_line_annotation(
    scene_with_helix_and_line: Scene,
    line_anno_A4_A6: LineAnnotation,
    mock_structure_coords: np.ndarray,
) -> None:
    """
    Tests if a LineAnnotation renders a line between expected coordinates.
    """
    scene = scene_with_helix_and_line
    renderer = SVGRenderer(scene=scene)

    svg_output = renderer.get_svg_string()

    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # --- Get ACTUAL base coordinates from element via resolver --- #
    resolver = scene.resolver
    base_coord_A4 = resolver.resolve(line_anno_A4_A6.start_coordinate)
    base_coord_A6 = resolver.resolve(line_anno_A4_A6.end_coordinate)
    assert base_coord_A4 is not None
    assert base_coord_A6 is not None

    # --- Calculate expected SVG positions --- #
    expected_x1 = base_coord_A4[0] + line_anno_A4_A6.style.offset[0]
    expected_y1 = base_coord_A4[1] + line_anno_A4_A6.style.offset[1]
    expected_x2 = base_coord_A6[0] + line_anno_A4_A6.style.offset[0]
    expected_y2 = base_coord_A6[1] + line_anno_A4_A6.style.offset[1]

    # Construct expected path data: "M x1,y1 L x2,y2"
    expected_d = (
        f"M{expected_x1:.1f},{expected_y1:.1f} L{expected_x2:.1f},{expected_y2:.1f}"
    )

    # Find the corresponding path element (was line)
    line_id = f"{line_anno_A4_A6.id}-line"
    paths = root.findall(
        f".//svg:path[@class='annotation line'][@id='{line_id}']", namespaces=ns
    )
    assert len(paths) == 1, f"Expected 1 annotation path (for line), found {len(paths)}"

    path = paths[0]
    actual_d = path.attrib.get("d", "")

    # Normalize and compare path data
    norm_actual_d = " ".join(actual_d.upper().replace(",", " ").split())
    norm_expected_d = " ".join(expected_d.upper().replace(",", " ").split())
    assert (
        norm_actual_d == norm_expected_d
    ), f"Expected path d='{norm_expected_d}', got '{norm_actual_d}'"

    # Check styling attributes
    assert path.attrib.get("stroke") == line_anno_A4_A6.style.line_color.as_hex()
    # Add checks for stroke-width, opacity etc. if needed

    # Check for connectors (still circles)
    connector_start_id = f"{line_anno_A4_A6.id}-connector-start"
    connector_end_id = f"{line_anno_A4_A6.id}-connector-end"
    conn_start = root.findall(f".//svg:circle[@id='{connector_start_id}']", ns)
    conn_end = root.findall(f".//svg:circle[@id='{connector_end_id}']", ns)
    assert len(conn_start) == 1, "Missing start connector circle"
    assert len(conn_end) == 1, "Missing end connector circle"
    # Verify connector positions match expected line endpoints
    assert float(conn_start[0].attrib.get("cx")) == pytest.approx(expected_x1)
    assert float(conn_start[0].attrib.get("cy")) == pytest.approx(expected_y1)
    assert float(conn_end[0].attrib.get("cx")) == pytest.approx(expected_x2)
    assert float(conn_end[0].attrib.get("cy")) == pytest.approx(expected_y2)


def test_render_area_annotation(
    scene_with_helix_and_area: Scene,
    area_anno_A5_A7: AreaAnnotation,
    mock_structure_coords: np.ndarray,
) -> None:
    """
    Tests if an AreaAnnotation renders a path matching the expected coordinates.
    """
    scene = scene_with_helix_and_area
    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # --- Get the ACTUAL coordinates the renderer will use --- #
    # These are the coordinates AFTER processing by AreaAnnotation.get_coordinates
    # Pass the resolver to the annotation's get_coordinates method
    processed_coords_3d = area_anno_A5_A7.get_coordinates(scene.resolver)
    assert processed_coords_3d is not None, "Annotation failed to calculate coordinates"
    assert (
        len(processed_coords_3d) >= 3
    ), f"Expected at least 3 processed points, got {len(processed_coords_3d)}"
    processed_coords_2d = processed_coords_3d[:, :2]  # Use only XY for path

    # Apply offset to the processed coordinates
    offset = area_anno_A5_A7.style.offset
    expected_path_points = processed_coords_2d + offset

    # Construct expected path data string from the *processed* points
    # expected_d = f"M{expected_path_points[0, 0]:.1f},{expected_path_points[0, 1]:.1f}"
    # for i in range(1, len(expected_path_points)):
    #     expected_d += f" L{expected_path_points[i, 0]:.1f},{expected_path_points[i, 1]:.1f}"
    # expected_d += " Z"

    # Find the corresponding path outline
    outline_id = f"{area_anno_A5_A7.id}-outline"
    paths = root.findall(
        f".//svg:path[@class='annotation area-outline'][@id='{outline_id}']",
        namespaces=ns,
    )
    assert len(paths) == 1, f"Expected 1 area outline path, found {len(paths)}"

    path = paths[0]
    actual_d = path.attrib.get("d", "")

    # Parse the actual path data
    actual_path_points = _parse_svg_path_d(actual_d)
    assert (
        actual_path_points is not None
    ), f"Could not parse actual path data: {actual_d}"

    # Compare the numerical coordinates with tolerance
    # Using a tolerance like 1e-1 because SVG rendering might round
    np.testing.assert_allclose(
        actual_path_points,
        expected_path_points,
        atol=1e-1,
        err_msg=f"Mismatch between expected points and rendered path points for {outline_id}",
    )

    # Check fill/stroke attributes based on _draw_area_annotation logic
    # Fill uses fill_color, Stroke uses base color
    assert (
        path.attrib.get("fill") == area_anno_A5_A7.style.fill_color.as_hex()
    ), f"Fill color mismatch: Expected {area_anno_A5_A7.style.fill_color.as_hex()}, Got {path.attrib.get('fill')}"
    assert (
        path.attrib.get("stroke") == area_anno_A5_A7.style.color.as_hex()
    ), f"Stroke color mismatch: Expected {area_anno_A5_A7.style.color.as_hex()}, Got {path.attrib.get('stroke')}"


# --- Phase 2 Tests ---


def test_render_point_with_custom_style(
    scene_with_helix: Scene,
    point_anno_A5: PointAnnotation,
    mock_structure_coords: np.ndarray,
) -> None:
    """Tests rendering a point annotation with non-default style attributes."""
    # Apply custom style
    custom_color = Color("#123456")
    custom_opacity = 0.6
    custom_radius = 8.0
    custom_offset = np.array([5.0, -5.0])
    point_anno_A5.style.color = custom_color
    point_anno_A5.style.opacity = custom_opacity
    point_anno_A5.style.marker_radius = custom_radius
    point_anno_A5.style.offset = custom_offset

    scene_with_helix.add_element(point_anno_A5)
    scene = scene_with_helix

    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # --- Get ACTUAL base coordinate from element via resolver --- #
    resolver = scene.resolver
    base_coords = resolver.resolve(point_anno_A5.target_coordinate)
    assert base_coords is not None

    # --- Calculate expected SVG position based on resolved coord and style --- #
    expected_cx = base_coords[0] + custom_offset[0]
    expected_cy = base_coords[1] + custom_offset[1]

    marker_id = f"{point_anno_A5.id}-marker"
    circles = root.findall(f".//svg:circle[@id='{marker_id}']", namespaces=ns)
    assert len(circles) == 1
    circle = circles[0]

    assert float(circle.attrib.get("cx")) == pytest.approx(expected_cx)
    assert float(circle.attrib.get("cy")) == pytest.approx(expected_cy)
    assert float(circle.attrib.get("r")) == pytest.approx(custom_radius)
    assert circle.attrib.get("fill").upper() == custom_color.as_hex().upper()
    assert float(circle.attrib.get("opacity")) == pytest.approx(custom_opacity)


def test_render_annotation_with_label(
    scene_with_helix: Scene,
    point_anno_A5: PointAnnotation,
    mock_structure_coords: np.ndarray,
) -> None:
    """Tests rendering an annotation with a label."""
    test_label = "My Point Label"
    custom_label_color = Color("#AA00BB")
    custom_font_size = 14.0
    point_anno_A5.label = test_label
    point_anno_A5.style.label_color = custom_label_color
    point_anno_A5.style.label_font_size = custom_font_size

    scene_with_helix.add_element(point_anno_A5)
    scene = scene_with_helix

    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Find the label text element
    label_id = f"{point_anno_A5.id}-label"
    labels = root.findall(
        f".//svg:text[@class='annotation label'][@id='{label_id}']", namespaces=ns
    )
    assert len(labels) == 1, f"Expected 1 label text element, found {len(labels)}"
    label = labels[0]

    assert label.text == test_label
    assert label.attrib.get("fill").upper() == custom_label_color.as_hex().upper()
    assert float(label.attrib.get("font-size")) == pytest.approx(custom_font_size)
    assert label.attrib.get("text-anchor") == "start"  # Default for point

    # Check position relative to marker (approximate check)
    marker_id = f"{point_anno_A5.id}-marker"
    marker = root.find(f".//svg:circle[@id='{marker_id}']", ns)
    assert marker is not None
    marker_cx = float(marker.attrib.get("cx"))
    label_x = float(label.attrib.get("x"))
    assert label_x > marker_cx  # Label should be to the right of the marker center


def test_render_invisible_annotation(
    scene_with_helix: Scene, point_anno_A5: PointAnnotation
) -> None:
    """Tests that an annotation with visibility=False is not rendered."""
    point_anno_A5.style.visibility = False
    scene_with_helix.add_element(point_anno_A5)
    scene = scene_with_helix

    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Assert that the marker and label (if any) are NOT found
    marker_id = f"{point_anno_A5.id}-marker"
    label_id = f"{point_anno_A5.id}-label"
    markers = root.findall(f".//svg:circle[@id='{marker_id}']", namespaces=ns)
    labels = root.findall(f".//svg:text[@id='{label_id}']", namespaces=ns)

    assert len(markers) == 0, "Invisible annotation marker should not be rendered."
    assert len(labels) == 0, "Invisible annotation label should not be rendered."
