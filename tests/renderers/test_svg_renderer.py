# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Callable
import pytest


import numpy as np
from pydantic_extra_types.color import Color
from xml.etree import ElementTree as ET

from flatprot.core import Structure, ResidueRangeSet, ResidueCoordinate
from flatprot.scene import (
    Scene,
    BaseSceneElement,
    SceneGroup,
    CoilSceneElement,
    HelixSceneElement,
    SheetSceneElement,
)
from flatprot.renderers import SVGRenderer

# Try importing the specific annotation type
try:
    from flatprot.scene.annotation.point import PointAnnotation
except ImportError:
    # Fallback to mocking BaseSceneElement if PointAnnotation isn't found
    PointAnnotation = BaseSceneElement


# --- Helper Functions ---


# Helper to parse SVG, potentially reuse from annotation tests later
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


# A more realistic mock structure, similar to test_scene_integration
@pytest.fixture
def mock_structure_coords() -> np.ndarray:
    """Provides coordinates for the mock structure."""
    # Simple linear coords for Chain A: 1-10
    coords = np.zeros((10, 3))
    coords[:, 0] = np.arange(10) * 5  # X coordinates increasing
    coords[:, 1] = 10  # Constant Y
    return coords


@pytest.fixture
def mock_structure(
    mock_structure_coords: np.ndarray, mocker
) -> Tuple[Structure, Callable]:
    """Provides a mock Structure object and its coordinate lookup function."""
    structure = mocker.MagicMock(spec=Structure)
    structure.id = "mock_struct_render"
    structure.coordinates = mock_structure_coords

    # Define the internal coordinate lookup logic
    def get_res_coord(
        res_coord: ResidueCoordinate,
        struct: Structure,  # Renamed arg for clarity
    ) -> Optional[np.ndarray]:
        # Use 'struct' argument consistently
        try:
            chain = struct.chains[res_coord.chain_id]
            idx = chain.coordinate_index(res_coord.residue_index)
            # Use struct.coordinates (the argument's attr)
            if 0 <= idx < len(struct.coordinates):
                return struct.coordinates[idx]
            return None
        except KeyError:
            return None

    # Mock chain 'A' (residues 1-10)
    mock_chain_a = mocker.MagicMock(name="ChainA")
    chain_a_coord_map = {i: i - 1 for i in range(1, 11)}

    def get_coord_index(residue_id: int) -> int:
        if residue_id in chain_a_coord_map:
            return chain_a_coord_map[residue_id]
        raise KeyError(f"Residue A:{residue_id} not found.")

    mock_chain_a.coordinate_index = mocker.MagicMock(side_effect=get_coord_index)
    mock_chain_a.__contains__ = mocker.MagicMock(
        side_effect=lambda res_idx: res_idx in chain_a_coord_map
    )
    mock_chain_a.id = "A"
    structure.chains = {"A": mock_chain_a}
    structure.__getitem__ = mocker.MagicMock(
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

    # Return both the mock object and the key function
    return structure, get_res_coord


@pytest.fixture
def empty_scene(mock_structure: Tuple[Structure, Callable]) -> Scene:
    """Provides an empty Scene object based on the mock structure."""
    structure_obj, _ = mock_structure  # Unpack, only need structure obj
    return Scene(structure=structure_obj)


# --- Structure Element Fixtures ---


@pytest.fixture
def coil_element() -> CoilSceneElement:
    """Provides a CoilSceneElement targeting Chain A, residues 1-5."""
    return CoilSceneElement(residue_range_set=ResidueRangeSet.from_string("A:1-5"))


@pytest.fixture
def helix_element() -> HelixSceneElement:
    """Provides a HelixSceneElement targeting Chain A, residues 3-7."""
    return HelixSceneElement(residue_range_set=ResidueRangeSet.from_string("A:3-7"))


@pytest.fixture
def sheet_element() -> SheetSceneElement:
    """Provides a SheetSceneElement targeting Chain A, residues 6-10."""
    return SheetSceneElement(residue_range_set=ResidueRangeSet.from_string("A:6-10"))


# --- Scene Fixtures with Elements ---


@pytest.fixture
def scene_with_coil(empty_scene: Scene, coil_element: CoilSceneElement) -> Scene:
    """Provides a Scene containing the coil element."""
    empty_scene.add_element(coil_element)
    return empty_scene


@pytest.fixture
def scene_with_helix(empty_scene: Scene, helix_element: HelixSceneElement) -> Scene:
    """Provides a Scene containing the helix element."""
    empty_scene.add_element(helix_element)
    return empty_scene


@pytest.fixture
def scene_with_sheet(empty_scene: Scene, sheet_element: SheetSceneElement) -> Scene:
    """Provides a Scene containing the sheet element."""
    empty_scene.add_element(sheet_element)
    return empty_scene


# --- Tests ---


def test_render_empty_scene(empty_scene: Scene) -> None:
    """
    Tests rendering an empty Scene produces valid SVG.
    """
    renderer = SVGRenderer(scene=empty_scene)  # Init renderer with the scene
    svg_output = renderer.get_svg_string()

    assert isinstance(svg_output, str)
    # Removed the startswith check - rely on parsing
    # assert svg_output.strip().lower().startswith("<svg")
    assert svg_output.strip().lower().endswith("</svg>")
    _parse_svg(svg_output)  # Check well-formedness


def test_render_coil(scene_with_coil: Scene, coil_element: CoilSceneElement) -> None:
    """
    Tests if a CoilSceneElement renders as an SVG path with correct class and attributes.
    """
    renderer = SVGRenderer(scene=scene_with_coil)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Use the automatically generated ID for lookup
    expected_id = coil_element.id  # Get the generated ID
    coil_paths = root.findall(
        f".//svg:path[@class='element coil'][@id='{expected_id}']", namespaces=ns
    )
    assert (
        len(coil_paths) == 1
    ), f"Expected 1 coil path with ID '{expected_id}', found {len(coil_paths)}"

    path = coil_paths[0]
    assert path.attrib.get("d"), "Coil path is missing 'd' attribute."
    assert (
        path.attrib.get("d", "").upper().startswith("M")
    ), "Path data should start with M"
    assert path.attrib.get("fill") == "none", "Coil path should not be filled"
    assert path.attrib.get("stroke"), "Coil path missing 'stroke' attribute"
    assert (
        float(path.attrib.get("stroke-width", 0)) > 0
    ), "Coil path has zero stroke width"


def test_render_helix(
    scene_with_helix: Scene,
    helix_element: HelixSceneElement,  # Removed mocker arg
) -> None:
    """
    Tests if a HelixSceneElement renders as a filled SVG path with correct class/attrs.
    """
    # Mocks are applied in the fixture now
    renderer = SVGRenderer(scene=scene_with_helix)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    expected_id = helix_element.id  # Get the generated ID
    helix_paths = root.findall(
        f".//svg:path[@class='element helix'][@id='{expected_id}']", namespaces=ns
    )
    assert (
        len(helix_paths) == 1
    ), f"Expected 1 helix path with ID '{expected_id}', found {len(helix_paths)}"

    path = helix_paths[0]
    d_attr = path.attrib.get("d", "")
    assert d_attr, "Helix path is missing 'd' attribute."
    assert d_attr.upper().startswith("M"), "Path data should start with M"
    assert d_attr.upper().endswith("Z"), "Helix path should be closed (end with Z)"
    assert path.attrib.get("fill") != "none", "Helix path should be filled"
    assert path.attrib.get("fill"), "Helix path missing 'fill' attribute"
    assert path.attrib.get("stroke"), "Helix path missing 'stroke' attribute"


def test_render_sheet(
    scene_with_sheet: Scene,
    sheet_element: SheetSceneElement,  # Removed mocker arg
) -> None:
    """
    Tests if a SheetSceneElement renders as a filled SVG path with correct class/attrs.
    """
    # Mocks are applied in the fixture now
    renderer = SVGRenderer(scene=scene_with_sheet)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    expected_id = sheet_element.id  # Get the generated ID
    sheet_paths = root.findall(
        f".//svg:path[@class='element sheet'][@id='{expected_id}']", namespaces=ns
    )
    assert (
        len(sheet_paths) == 1
    ), f"Expected 1 sheet path with ID '{expected_id}', found {len(sheet_paths)}"

    path = sheet_paths[0]
    d_attr = path.attrib.get("d", "")
    assert d_attr, "Sheet path is missing 'd' attribute."
    assert d_attr.upper().startswith("M"), "Path data should start with M"
    assert d_attr.upper().endswith("Z"), "Sheet path should be closed (end with Z)"
    assert path.attrib.get("fill") != "none", "Sheet path should be filled"
    assert path.attrib.get("fill"), "Sheet path missing 'fill' attribute"
    assert path.attrib.get("stroke"), "Sheet path missing 'stroke' attribute"


# --- Phase 2 Tests ---


def test_render_helix_with_custom_style(
    empty_scene: Scene,
    helix_element: HelixSceneElement,  # Removed mocker arg
) -> None:
    """Tests rendering a helix with non-default style attributes."""
    # Apply custom style
    custom_fill = Color("#FF0000")  # Red
    custom_stroke = Color("#00FF00")  # Green
    custom_stroke_width = 3.5
    custom_opacity = 0.75
    helix_element.style.color = custom_fill
    helix_element.style.stroke_color = custom_stroke
    helix_element.style.stroke_width = custom_stroke_width
    helix_element.style.opacity = custom_opacity

    empty_scene.add_element(helix_element)
    scene = empty_scene

    # Mocks applied in fixture
    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    expected_id = helix_element.id  # Get generated ID
    helix_paths = root.findall(
        f".//svg:path[@class='element helix'][@id='{expected_id}']", namespaces=ns
    )
    assert len(helix_paths) == 1

    path = helix_paths[0]
    assert path.attrib.get("fill").upper() == custom_fill.as_hex().upper()
    assert path.attrib.get("stroke").upper() == custom_stroke.as_hex().upper()
    assert float(path.attrib.get("stroke-width")) == pytest.approx(custom_stroke_width)
    # fill-opacity and stroke-opacity are derived from main opacity in _draw_helix
    assert float(path.attrib.get("fill-opacity")) == pytest.approx(custom_opacity)
    assert float(path.attrib.get("stroke-opacity")) == pytest.approx(custom_opacity)


def test_render_invisible_element(
    empty_scene: Scene, coil_element: CoilSceneElement
) -> None:
    """Tests that an element with visibility=False is not rendered."""
    coil_element.style.visibility = False
    empty_scene.add_element(coil_element)
    scene = empty_scene

    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    expected_id = coil_element.id  # Get generated ID
    # Assert that the path for this element is NOT found
    coil_paths = root.findall(f".//svg:path[@id='{expected_id}']", namespaces=ns)
    assert len(coil_paths) == 0, "Invisible element should not be rendered."


def test_render_background(empty_scene: Scene) -> None:
    """Tests background rendering with color and transparency."""
    custom_bg = "#ABCDEF"
    custom_opacity = 0.5

    # Test with color
    renderer_color = SVGRenderer(
        scene=empty_scene, background_color=custom_bg, background_opacity=custom_opacity
    )
    svg_output_color = renderer_color.get_svg_string()
    root_color = _parse_svg(svg_output_color)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    bg_rects = root_color.findall(".//svg:rect[@class='background']", namespaces=ns)
    assert len(bg_rects) == 1, "Expected one background rectangle"
    bg_rect = bg_rects[0]
    assert bg_rect.attrib.get("fill").upper() == custom_bg.upper()
    assert float(bg_rect.attrib.get("opacity")) == pytest.approx(custom_opacity)

    # Test with no background
    renderer_none = SVGRenderer(scene=empty_scene, background_color=None)
    svg_output_none = renderer_none.get_svg_string()
    root_none = _parse_svg(svg_output_none)
    bg_rects_none = root_none.findall(".//svg:rect[@class='background']", namespaces=ns)
    assert (
        len(bg_rects_none) == 0
    ), "Expected no background rectangle when color is None"


def test_render_custom_dimensions(empty_scene: Scene) -> None:
    """Tests rendering with custom width and height."""
    custom_width = 850
    custom_height = 550
    renderer = SVGRenderer(scene=empty_scene, width=custom_width, height=custom_height)
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)

    assert root.tag.endswith("svg")  # Basic check it's the root svg tag
    assert root.attrib.get("width") == str(custom_width)
    assert root.attrib.get("height") == str(custom_height)


def test_render_with_scenegroup(empty_scene: Scene) -> None:
    """Tests rendering an element nested within a SceneGroup."""
    group_id = "my_test_group"
    empty_scene.add_element(SceneGroup(id=group_id))

    # Create coil element *locally* within the test
    coil_element = CoilSceneElement(
        residue_range_set=ResidueRangeSet.from_string("A:1-5")
    )
    empty_scene.add_element(coil_element, parent_id=group_id)

    # Add the locally created coil to the group
    scene = empty_scene

    renderer = SVGRenderer(scene=scene)
    svg_output = renderer.get_svg_string()

    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Find the SVG group
    svg_groups = root.findall(f".//svg:g[@id='{group_id}']", namespaces=ns)
    assert len(svg_groups) == 1, f"Expected SVG group with id '{group_id}'"
    svg_group = svg_groups[0]

    # Get the expected ID (will be generated based on local instance)
    expected_coil_id = coil_element.id

    # Check: Does the path exist ANYWHERE in the document?
    coil_paths_anywhere = root.findall(
        f".//svg:path[@id='{expected_coil_id}']", namespaces=ns
    )
    assert (
        len(coil_paths_anywhere) == 1
    ), f"Coil path with ID '{expected_coil_id}' not found anywhere in the SVG."

    # Original check: Find the coil path *within* the group
    coil_paths_in_group = svg_group.findall(
        f"./svg:path[@id='{expected_coil_id}']",
        namespaces=ns,  # Note ./ for relative path
    )
    assert len(coil_paths_in_group) == 1, "Expected coil path within the SVG group"

    # Verify coil path is not at the root
    coil_paths_at_root = root.findall(
        f"./svg:path[@id='{expected_coil_id}']",
        namespaces=ns,  # Note ./ for relative path
    )
    assert len(coil_paths_at_root) == 0, "Coil path should not be at the SVG root"


def test_render_structure_coord_error(empty_scene: Scene) -> None:
    """Tests rendering when a structure element cannot get coordinates."""
    # Target a residue that doesn't exist in the mock structure
    # ID is generated automatically
    bad_coil = CoilSceneElement(
        residue_range_set=ResidueRangeSet.from_string("A:11-15")
    )
    empty_scene.add_element(bad_coil)
    scene = empty_scene

    renderer = SVGRenderer(scene=scene)
    # Should ideally log a warning, but rendering shouldn't fail
    svg_output = renderer.get_svg_string()
    root = _parse_svg(svg_output)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    expected_bad_id = bad_coil.id  # Get the generated ID
    # Assert that the path for the bad coil is NOT found
    bad_coil_paths = root.findall(
        f".//svg:path[@id='{expected_bad_id}']", namespaces=ns
    )
    assert (
        len(bad_coil_paths) == 0
    ), "Element with coordinate error should not be rendered."
