# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from xml.etree import ElementTree as ET
from typing import Optional, Tuple, Callable

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


# --- Fixtures ---


# Reusing mock structure setup from test_svg_renderer
@pytest.fixture
def mock_structure_coords() -> np.ndarray:
    """Provides coordinates for the mock structure."""
    coords = np.zeros((10, 3))
    coords[:, 0] = np.arange(10) * 5  # X coordinates: 0, 5, 10, ..., 45
    coords[:, 1] = 10  # Constant Y
    return coords


@pytest.fixture
def mock_structure(
    mock_structure_coords: np.ndarray, mocker
) -> Tuple[Structure, Callable]:
    """Provides a mock Structure object and its coordinate lookup function."""
    structure = mocker.MagicMock(spec=Structure)
    structure.id = "mock_struct_anno"
    structure.coordinates = mock_structure_coords

    # Define the internal coordinate lookup logic
    def get_res_coord(
        res_coord: ResidueCoordinate, struct: Structure
    ) -> Optional[np.ndarray]:
        # IMPORTANT: Use struct argument consistently for lookups
        try:
            chain = struct.chains[res_coord.chain_id]
            idx = chain.coordinate_index(res_coord.residue_index)
            # Use struct.coordinates for bounds check and retrieval
            if 0 <= idx < len(struct.coordinates):
                return struct.coordinates[idx]
            return None
        except KeyError:
            return None

    # Mock chain 'A' (residues 1-10)
    mock_chain_a = mocker.MagicMock(name="ChainA")
    chain_a_coord_map = {i: i - 1 for i in range(1, 11)}

    def get_coord_index(residue_id: int) -> int:
        # Simplified: chain_id check happens in get_res_coord
        if residue_id in chain_a_coord_map:
            return chain_a_coord_map[residue_id]
        raise KeyError(f"Residue A:{residue_id} not found.")

    mock_chain_a.coordinate_index = mocker.MagicMock(side_effect=get_coord_index)
    mock_chain_a.__contains__ = mocker.MagicMock(  # Mock the 'in' operator check
        side_effect=lambda res_idx: res_idx in chain_a_coord_map
    )
    mock_chain_a.id = "A"
    structure.chains = {"A": mock_chain_a}
    structure.__getitem__ = mocker.MagicMock(  # Mock structure['A']
        side_effect=lambda chain_id: structure.chains[chain_id]
    )

    # Assign the mock method using the actual function
    structure.get_residue_coordinate = mocker.MagicMock(side_effect=get_res_coord)

    structure.get_residues_in_range = mocker.MagicMock(
        side_effect=lambda rrs: list(rrs.residues)
    )
    structure.calculate_helix_sheet_coords = mocker.MagicMock(
        return_value=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    )
    # This mock might be unnecessary now if get_coordinate_at_residue uses the base implementation
    structure.get_coordinate_at_residue_in_element = mocker.MagicMock(
        side_effect=get_res_coord
    )

    # Return both the mock object and the key function
    return structure, get_res_coord


@pytest.fixture
def empty_scene(mock_structure: Tuple[Structure, Callable]) -> Scene:
    """Provides an empty Scene object based on the mock structure."""
    structure_obj, _ = mock_structure  # Unpack, only need structure obj
    return Scene(structure=structure_obj)


# --- Structure Element Fixture (for annotations to target) ---


@pytest.fixture
def helix_A_3_7(
    mock_structure: Tuple[Structure, Callable], mocker
) -> HelixSceneElement:
    """Provides a HelixSceneElement targeting Chain A, residues 3-7."""
    structure_obj, get_res_coord_func = mock_structure  # Unpack
    # ID is generated automatically
    helix = HelixSceneElement(residue_range_set=ResidueRangeSet.from_string("A:3-7"))

    # Mock the element's method, make side_effect call the actual lookup function directly
    def mock_side_effect(
        residue_coordinate: ResidueCoordinate, structure_arg: Structure
    ):
        # Call the actual lookup function directly, passing the structure arg received
        return get_res_coord_func(residue_coordinate, structure_arg)

    helix.get_coordinate_at_residue = mocker.MagicMock(side_effect=mock_side_effect)
    return helix


# --- Annotation Fixtures (targeting residues) ---


@pytest.fixture
def point_anno_A5() -> PointAnnotation:
    """Provides a PointAnnotation targeting residue A:5."""
    return PointAnnotation(
        id="point_on_A5", target_coordinate=ResidueCoordinate("A", 5)
    )


@pytest.fixture
def line_anno_A4_A6() -> LineAnnotation:
    """Provides a LineAnnotation targeting residues A:4 and A:6."""
    return LineAnnotation(
        id="line_A4_A6",
        target_coordinates=[
            ResidueCoordinate("A", 4),
            ResidueCoordinate("A", 6),
        ],
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
    empty_scene.add_node(helix_A_3_7)
    return empty_scene


@pytest.fixture
def scene_with_helix_and_point(
    scene_with_helix: Scene, point_anno_A5: PointAnnotation
) -> Scene:
    """Scene with helix A:3-7 and a point annotation on A:5."""
    scene_with_helix.add_node(point_anno_A5)
    return scene_with_helix


@pytest.fixture
def scene_with_helix_and_line(
    scene_with_helix: Scene, line_anno_A4_A6: LineAnnotation
) -> Scene:
    """Scene with helix A:3-7 and a line annotation from A:4 to A:6."""
    scene_with_helix.add_node(line_anno_A4_A6)
    return scene_with_helix


@pytest.fixture
def scene_with_helix_and_area(
    scene_with_helix: Scene, area_anno_A5_A7: AreaAnnotation
) -> Scene:
    """Scene with helix A:3-7 and an area annotation covering A:5-7."""
    scene_with_helix.add_node(area_anno_A5_A7)
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

    # Expected coordinates for A:5 (index 4)
    expected_coords = mock_structure_coords[4]  # Should be [20., 10., 0.]
    expected_cx = expected_coords[0] + point_anno_A5.style.offset[0]
    expected_cy = expected_coords[1] + point_anno_A5.style.offset[1]

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

    # Expected coordinates for A:4 (index 3) and A:6 (index 5)
    coord_A4 = mock_structure_coords[3]  # [15., 10., 0.]
    coord_A6 = mock_structure_coords[5]  # [25., 10., 0.]

    expected_x1 = coord_A4[0] + line_anno_A4_A6.style.offset[0]
    expected_y1 = coord_A4[1] + line_anno_A4_A6.style.offset[1]
    expected_x2 = coord_A6[0] + line_anno_A4_A6.style.offset[0]
    expected_y2 = coord_A6[1] + line_anno_A4_A6.style.offset[1]

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
    assert path.attrib.get("stroke") == line_anno_A4_A6.style.color.as_hex()
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

    # Expected coordinates for A:5, A:6, A:7 (indices 4, 5, 6)
    coord_A5 = mock_structure_coords[4]  # [20., 10., 0.]
    coord_A6 = mock_structure_coords[5]  # [25., 10., 0.]
    coord_A7 = mock_structure_coords[6]  # [30., 10., 0.]
    expected_raw_coords = np.array([coord_A5, coord_A6, coord_A7])

    # Apply offset
    offset = area_anno_A5_A7.style.offset
    expected_offset_coords = expected_raw_coords[:, :2] + offset

    # Construct expected path data string (M x0,y0 L x1,y1 L x2,y2 Z)
    # Use f-string formatting with precision to avoid minor float issues
    expected_d = (
        f"M{expected_offset_coords[0, 0]:.1f},{expected_offset_coords[0, 1]:.1f}"
    )
    for i in range(1, len(expected_offset_coords)):
        expected_d += (
            f" L{expected_offset_coords[i, 0]:.1f},{expected_offset_coords[i, 1]:.1f}"
        )
    expected_d += " Z"

    # Find the corresponding path outline
    outline_id = f"{area_anno_A5_A7.id}-outline"
    paths = root.findall(
        f".//svg:path[@class='annotation area-outline'][@id='{outline_id}']",
        namespaces=ns,
    )
    assert len(paths) == 1, f"Expected 1 area outline path, found {len(paths)}"

    path = paths[0]
    actual_d = path.attrib.get("d", "")

    # Normalize path data (uppercase, spacing) for robust comparison
    norm_actual_d = " ".join(actual_d.upper().replace(",", " ").split())
    norm_expected_d = " ".join(expected_d.upper().replace(",", " ").split())

    assert (
        norm_actual_d == norm_expected_d
    ), f"Expected path d='{norm_expected_d}', got '{norm_actual_d}'"
    assert path.attrib.get("fill") == area_anno_A5_A7.style.color.as_hex()
    assert path.attrib.get("stroke") == area_anno_A5_A7.style.color.as_hex()


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

    scene_with_helix.add_node(point_anno_A5)
    scene = scene_with_helix

    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Expected base coordinates for A:5 (index 4)
    expected_coords = mock_structure_coords[4]  # [20., 10., 0.]
    expected_cx = expected_coords[0] + custom_offset[0]
    expected_cy = expected_coords[1] + custom_offset[1]

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

    scene_with_helix.add_node(point_anno_A5)
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
    scene_with_helix.add_node(point_anno_A5)
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
