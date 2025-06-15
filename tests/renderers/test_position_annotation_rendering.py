# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for position annotation SVG rendering."""

from xml.etree import ElementTree as ET
from typing import Optional

import pytest
import numpy as np

from flatprot.core import (
    Structure,
    ResidueCoordinate,
    ResidueRange,
)
from flatprot.scene import (
    Scene,
    PositionAnnotation,
    PositionAnnotationStyle,
    PositionType,
    HelixSceneElement,
)
from flatprot.renderers import SVGRenderer
from flatprot.renderers.svg_annotations import _draw_position_annotation


# --- Helper Functions ---


def _parse_svg(svg_output: str) -> ET.Element:
    """Helper to parse SVG and fail test on error."""
    try:
        if not svg_output or not svg_output.strip():
            pytest.fail("Generated SVG output is empty or whitespace only.")
        return ET.fromstring(svg_output)
    except ET.ParseError as e:
        pytest.fail(f"Generated SVG is not well-formed XML: {e}\nOutput:\n{svg_output}")


# --- Fixtures ---


@pytest.fixture
def mock_structure_coords() -> np.ndarray:
    """Provides coordinates for the mock structure."""
    coords = np.zeros((10, 3))
    coords[:, 0] = np.arange(10) * 5  # X coordinates: 0, 5, 10, ..., 45
    coords[:, 1] = 10  # Constant Y
    coords[:, 2] = 0  # Z depth
    return coords


@pytest.fixture
def mock_structure(mock_structure_coords: np.ndarray, mocker) -> Structure:
    """Provides a mock Structure object."""
    structure = mocker.MagicMock(spec=Structure)
    structure.id = "mock_struct_pos"
    structure.coordinates = mock_structure_coords

    # Mock Chain A
    mock_chain_a = mocker.MagicMock(name="ChainA")
    chain_a_coord_map = {i: i - 1 for i in range(1, 11)}

    def coord_index_a(res_idx):
        idx = chain_a_coord_map.get(res_idx)
        if idx is None:
            raise KeyError(f"Simulated key error for A:{res_idx}")
        return idx

    mock_chain_a.coordinate_index.side_effect = coord_index_a
    mock_chain_a.__contains__.side_effect = lambda res_idx: res_idx in chain_a_coord_map
    mock_chain_a.id = "A"

    def getitem_side_effect(chain_id):
        if chain_id == "A":
            return mock_chain_a
        else:
            raise KeyError(f"Simulated chain {chain_id} not found")

    structure.__getitem__.side_effect = getitem_side_effect
    structure.chains = {"A": mock_chain_a}

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
    """Provides an empty Scene object."""
    return Scene(structure=mock_structure)


@pytest.fixture
def position_annotation_n_terminus() -> PositionAnnotation:
    """Provides an N-terminus position annotation."""
    return PositionAnnotation(
        id="n_terminus_pos",
        target=ResidueCoordinate("A", 1, None, 0),
        position_type=PositionType.N_TERMINUS,
        text="N",
    )


@pytest.fixture
def position_annotation_c_terminus() -> PositionAnnotation:
    """Provides a C-terminus position annotation."""
    return PositionAnnotation(
        id="c_terminus_pos",
        target=ResidueCoordinate("A", 10, None, 9),
        position_type=PositionType.C_TERMINUS,
        text="C",
    )


@pytest.fixture
def position_annotation_residue_number() -> PositionAnnotation:
    """Provides a residue number position annotation."""
    return PositionAnnotation(
        id="residue_42_pos",
        target=ResidueRange("A", 42, 48, 41, None),
        position_type=PositionType.RESIDUE_NUMBER,
        text="42",
    )


@pytest.fixture
def helix_element_covering_residues(mock_structure: Structure) -> HelixSceneElement:
    """Provides a helix element that covers residues 1-10."""
    from flatprot.core import ResidueRangeSet

    helix = HelixSceneElement(residue_range_set=ResidueRangeSet.from_string("A:1-10"))
    return helix


@pytest.fixture
def scene_with_position_annotation(
    empty_scene: Scene,
    position_annotation_n_terminus: PositionAnnotation,
    helix_element_covering_residues: HelixSceneElement,
) -> Scene:
    """Scene with a position annotation and structure element to resolve coordinates."""
    # Add structure element first so position annotation can resolve coordinates
    empty_scene.add_element(helix_element_covering_residues)
    empty_scene.add_element(position_annotation_n_terminus)
    return empty_scene


# --- Direct Function Tests ---


class TestDrawPositionAnnotation:
    """Test the _draw_position_annotation function directly."""

    def test_draw_position_annotation_basic(self, position_annotation_n_terminus):
        """Test basic position annotation drawing."""
        anchor_coords = np.array([[10.0, 20.0, 5.0]])

        result = _draw_position_annotation(
            position_annotation_n_terminus, anchor_coords
        )

        # Just check that we get a Text object back
        assert result is not None

    def test_draw_position_annotation_with_offset(
        self, position_annotation_residue_number
    ):
        """Test position annotation drawing with offset."""
        anchor_coords = np.array([[15.0, 25.0, 8.0]])

        result = _draw_position_annotation(
            position_annotation_residue_number, anchor_coords
        )

        assert result is not None

    def test_draw_position_annotation_custom_style(
        self, position_annotation_c_terminus
    ):
        """Test position annotation with custom styling."""
        anchor_coords = np.array([[30.0, 40.0, 10.0]])

        result = _draw_position_annotation(
            position_annotation_c_terminus, anchor_coords
        )

        assert result is not None

    def test_draw_position_annotation_no_coordinates(
        self, position_annotation_n_terminus
    ):
        """Test position annotation with no anchor coordinates."""
        result = _draw_position_annotation(position_annotation_n_terminus, None)
        assert result is None

    def test_draw_position_annotation_empty_coordinates(
        self, position_annotation_n_terminus
    ):
        """Test position annotation with empty coordinate array."""
        empty_coords = np.array([]).reshape(0, 3)
        result = _draw_position_annotation(position_annotation_n_terminus, empty_coords)
        assert result is None


# --- Integration Tests ---


class TestPositionAnnotationRendering:
    """Test position annotation rendering in full SVG context."""

    def test_render_n_terminus_annotation(
        self,
        scene_with_position_annotation: Scene,
        position_annotation_n_terminus: PositionAnnotation,
    ):
        """Test rendering N-terminus annotation in SVG."""
        scene = scene_with_position_annotation
        renderer = SVGRenderer(scene=scene)
        svg_output = renderer.get_svg_string()
        root = _parse_svg(svg_output)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        # Find the position annotation text
        text_elements = root.findall(
            f".//svg:text[@class='annotation position-annotation'][@id='{position_annotation_n_terminus.id}']",
            namespaces=ns,
        )
        assert (
            len(text_elements) == 1
        ), f"Expected 1 position annotation text, found {len(text_elements)}"

        text_elem = text_elements[0]
        assert text_elem.text == "N"

    def test_render_multiple_position_annotations(
        self, empty_scene: Scene, helix_element_covering_residues: HelixSceneElement
    ):
        """Test rendering multiple position annotations."""
        # Add structure element to cover residues
        empty_scene.add_element(helix_element_covering_residues)

        # Add N-terminus
        n_term = PositionAnnotation(
            id="n_term",
            target=ResidueCoordinate("A", 1, None, 0),
            position_type=PositionType.N_TERMINUS,
            text="N",
        )
        empty_scene.add_element(n_term)

        # Add C-terminus
        c_term = PositionAnnotation(
            id="c_term",
            target=ResidueCoordinate("A", 10, None, 9),
            position_type=PositionType.C_TERMINUS,
            text="C",
        )
        empty_scene.add_element(c_term)

        # Add residue number
        res_num = PositionAnnotation(
            id="res_42",
            target=ResidueCoordinate("A", 5, None, 4),
            position_type=PositionType.RESIDUE_NUMBER,
            text="42",
        )
        empty_scene.add_element(res_num)

        renderer = SVGRenderer(scene=empty_scene)
        svg_output = renderer.get_svg_string()
        root = _parse_svg(svg_output)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        # Should find all three position annotations
        text_elements = root.findall(
            ".//svg:text[@class='annotation position-annotation']",
            namespaces=ns,
        )
        assert len(text_elements) == 3

        # Check each annotation is present
        texts = [elem.text for elem in text_elements]
        assert "N" in texts
        assert "C" in texts
        assert "42" in texts

    def test_render_position_annotation_with_invisible_style(self, empty_scene: Scene):
        """Test that invisible position annotations are not rendered."""
        annotation = PositionAnnotation(
            id="invisible_pos",
            target=ResidueCoordinate("A", 5, None, 4),
            position_type=PositionType.RESIDUE_NUMBER,
            text="42",
        )
        annotation.style.visibility = False
        empty_scene.add_element(annotation)

        renderer = SVGRenderer(scene=empty_scene)
        svg_output = renderer.get_svg_string()
        root = _parse_svg(svg_output)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        # Should not find any position annotation text
        text_elements = root.findall(
            f".//svg:text[@id='{annotation.id}']",
            namespaces=ns,
        )
        assert (
            len(text_elements) == 0
        ), "Invisible position annotation should not be rendered"

    def test_position_annotation_font_properties(
        self, empty_scene: Scene, helix_element_covering_residues: HelixSceneElement
    ):
        """Test that position annotation font properties are correctly applied."""
        # Add structure element to cover residues
        empty_scene.add_element(helix_element_covering_residues)

        style = PositionAnnotationStyle(
            terminus_font_size=16.0,
            terminus_font_weight="bold",
            font_family="Times New Roman",
        )

        annotation = PositionAnnotation(
            id="styled_n_term",
            target=ResidueCoordinate("A", 1, None, 0),
            position_type=PositionType.N_TERMINUS,
            text="N",
            style=style,
        )
        empty_scene.add_element(annotation)

        renderer = SVGRenderer(scene=empty_scene)
        svg_output = renderer.get_svg_string()
        root = _parse_svg(svg_output)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        text_elements = root.findall(
            f".//svg:text[@id='{annotation.id}']",
            namespaces=ns,
        )
        assert len(text_elements) == 1

        text_elem = text_elements[0]
        assert text_elem.text == "N"
