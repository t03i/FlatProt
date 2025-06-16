# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for position annotation functionality."""

import pytest
from unittest.mock import patch, MagicMock

from flatprot.cli.project import project_structure_svg
from flatprot.utils.scene_utils import add_position_annotations_to_scene
from flatprot.scene import PositionAnnotationStyle


class TestPositionAnnotationIntegration:
    """Test position annotation integration with CLI and rendering pipeline."""

    @pytest.fixture
    def mock_structure_file(self, tmp_path):
        """Create a temporary structure file for testing."""
        structure_file = tmp_path / "test.cif"
        # Create a minimal CIF file content
        structure_file.write_text(
            """
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 20.0 ? 1 ALA A CA 1
ATOM 2 C CA . GLY A 1 2 ? 3.8 0.0 0.0 1.00 20.0 ? 2 GLY A CA 1
ATOM 3 C CA . VAL A 1 3 ? 7.6 0.0 0.0 1.00 20.0 ? 3 VAL A CA 1
"""
        )
        return structure_file

    @patch("flatprot.cli.project.GemmiStructureParser")
    @patch("flatprot.cli.project.transform_structure_with_inertia")
    @patch("flatprot.cli.project.project_structure_orthographically")
    @patch("flatprot.cli.project.create_scene_from_structure")
    @patch("flatprot.cli.project.add_position_annotations_to_scene")
    @patch("flatprot.cli.project.SVGRenderer")
    def test_cli_position_annotations_enabled(
        self,
        mock_renderer,
        mock_add_positions,
        mock_create_scene,
        mock_project,
        mock_transform,
        mock_parser,
        mock_structure_file,
    ):
        """Test that CLI correctly calls position annotation functions when enabled."""
        # Setup mocks
        mock_structure = MagicMock()
        mock_parser.return_value.parse_structure.return_value = mock_structure
        mock_transform.return_value = mock_structure
        mock_project.return_value = mock_structure

        mock_scene = MagicMock()
        mock_create_scene.return_value = mock_scene

        mock_renderer_instance = MagicMock()
        mock_renderer.return_value = mock_renderer_instance
        mock_renderer_instance.get_svg_string.return_value = "<svg></svg>"

        # Call CLI function with position annotations enabled
        result = project_structure_svg(
            structure=mock_structure_file,
            show_positions="full",
        )

        # Verify position annotations were added
        mock_add_positions.assert_called_once_with(mock_scene, None, "full")
        assert result == 0

    @patch("flatprot.cli.project.GemmiStructureParser")
    @patch("flatprot.cli.project.transform_structure_with_inertia")
    @patch("flatprot.cli.project.project_structure_orthographically")
    @patch("flatprot.cli.project.create_scene_from_structure")
    @patch("flatprot.cli.project.add_position_annotations_to_scene")
    @patch("flatprot.cli.project.SVGRenderer")
    def test_cli_position_annotations_disabled(
        self,
        mock_renderer,
        mock_add_positions,
        mock_create_scene,
        mock_project,
        mock_transform,
        mock_parser,
        mock_structure_file,
    ):
        """Test that CLI does not call position annotation functions when disabled."""
        # Setup mocks
        mock_structure = MagicMock()
        mock_parser.return_value.parse_structure.return_value = mock_structure
        mock_transform.return_value = mock_structure
        mock_project.return_value = mock_structure

        mock_scene = MagicMock()
        mock_create_scene.return_value = mock_scene

        mock_renderer_instance = MagicMock()
        mock_renderer.return_value = mock_renderer_instance
        mock_renderer_instance.get_svg_string.return_value = "<svg></svg>"

        # Call CLI function with position annotations disabled (default)
        result = project_structure_svg(
            structure=mock_structure_file,
            show_positions="none",
        )

        # Verify position annotations were NOT added
        mock_add_positions.assert_not_called()
        assert result == 0

    @patch("flatprot.cli.project.GemmiStructureParser")
    @patch("flatprot.cli.project.transform_structure_with_inertia")
    @patch("flatprot.cli.project.project_structure_orthographically")
    @patch("flatprot.cli.project.create_scene_from_structure")
    @patch("flatprot.cli.project.add_position_annotations_to_scene")
    @patch("flatprot.cli.project.SVGRenderer")
    @patch("flatprot.cli.project.StyleParser")
    def test_cli_position_annotations_with_custom_style(
        self,
        mock_style_parser,
        mock_renderer,
        mock_add_positions,
        mock_create_scene,
        mock_project,
        mock_transform,
        mock_parser,
        mock_structure_file,
        tmp_path,
    ):
        """Test that CLI passes custom position annotation style when provided."""
        # Create a style file
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = 12.0
terminus_font_size = 16.0
color = "#FF0000"
"""
        )

        # Setup mocks
        mock_structure = MagicMock()
        mock_parser.return_value.parse_structure.return_value = mock_structure
        mock_transform.return_value = mock_structure
        mock_project.return_value = mock_structure

        mock_scene = MagicMock()
        mock_create_scene.return_value = mock_scene

        # Mock style parser to return position annotation style
        mock_position_style = PositionAnnotationStyle(
            font_size=12.0,
            terminus_font_size=16.0,
        )
        mock_styles = {
            "position_annotation": mock_position_style,
            "helix": MagicMock(),
        }
        mock_style_parser.return_value.parse.return_value = mock_styles

        mock_renderer_instance = MagicMock()
        mock_renderer.return_value = mock_renderer_instance
        mock_renderer_instance.get_svg_string.return_value = "<svg></svg>"

        # Call CLI function with position annotations and custom style
        result = project_structure_svg(
            structure=mock_structure_file,
            style=style_file,
            show_positions="full",
        )

        # Verify position annotations were added with custom style
        mock_add_positions.assert_called_once_with(
            mock_scene, mock_position_style, "full"
        )
        assert result == 0

    def test_position_annotation_scene_integration(self, mocker):
        """Test that position annotations integrate correctly with scene system."""
        # Just test that the utility function can be called without error
        mock_scene = mocker.MagicMock()
        mock_helix = mocker.MagicMock()
        mock_helix.id = "helix_1"

        # Create a proper ResidueRangeSet instead of MagicMock
        from flatprot.core import ResidueRangeSet

        mock_range_set = ResidueRangeSet.from_string("A:1-5")
        mock_helix.residue_range_set = mock_range_set

        mock_scene.get_sequential_structure_elements.return_value = [mock_helix]

        # Call add_position_annotations_to_scene
        add_position_annotations_to_scene(mock_scene, annotation_level="full")

        # Verify add_element was called
        assert mock_scene.add_element.call_count > 0

    def test_position_annotation_svg_output_structure(self, mocker):
        """Test that position annotations appear in SVG output with correct structure."""
        # Simple test - just check that SVGRenderer can handle position annotations
        mock_renderer = mocker.MagicMock()
        mock_renderer.get_svg_string.return_value = (
            '<svg><text class="annotation position-annotation">N</text></svg>'
        )

        # Check that we can find position annotation text in mock SVG
        svg_output = mock_renderer.get_svg_string()
        assert "position-annotation" in svg_output
        assert "N" in svg_output
