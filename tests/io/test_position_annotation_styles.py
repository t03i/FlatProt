# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for position annotation style system integration."""

import pytest
from pydantic_extra_types.color import Color

from flatprot.io.styles import StyleParser
from flatprot.scene.annotation import PositionAnnotationStyle


class TestPositionAnnotationStyleParsing:
    """Test position annotation style parsing from TOML files."""

    def test_parse_position_annotation_style_basic(self, tmp_path):
        """Test parsing basic position annotation style from TOML."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = 10.0
terminus_font_size = 14.0
font_weight = "bold"
font_family = "Helvetica"
text_offset = 5.0
show_terminus = false
show_residue_numbers = true
terminus_font_weight = "normal"
color = "#FF0000"
opacity = 0.8
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        assert "position_annotation" in styles
        pos_style = styles["position_annotation"]
        assert isinstance(pos_style, PositionAnnotationStyle)

        assert pos_style.font_size == 10.0
        assert pos_style.terminus_font_size == 14.0
        assert pos_style.font_weight == "bold"
        assert pos_style.font_family == "Helvetica"
        assert pos_style.text_offset == 5.0
        assert pos_style.show_terminus is False
        assert pos_style.show_residue_numbers is True
        assert pos_style.terminus_font_weight == "normal"
        assert pos_style.color == Color("#FF0000")
        assert pos_style.opacity == 0.8

    def test_parse_position_annotation_style_minimal(self, tmp_path):
        """Test parsing minimal position annotation style (using defaults)."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = 12.0
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        assert "position_annotation" in styles
        pos_style = styles["position_annotation"]

        # Check custom value
        assert pos_style.font_size == 12.0

        # Check defaults are preserved
        assert pos_style.terminus_font_size == 10.0  # Default
        assert pos_style.font_weight == "normal"  # Default
        assert pos_style.show_terminus is True  # Default

    def test_parse_mixed_styles_with_position_annotation(self, tmp_path):
        """Test parsing multiple style sections including position annotations."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[helix]
color = "#FF0000"

[sheet]
color = "#0000FF"

[position_annotation]
font_size = 11.0
terminus_font_size = 15.0
color = "#00FF00"

[connection]
color = "#FFFF00"
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        # Should have all sections
        assert "helix" in styles
        assert "sheet" in styles
        assert "position_annotation" in styles
        assert "connection" in styles

        # Check position annotation specifically
        pos_style = styles["position_annotation"]
        assert isinstance(pos_style, PositionAnnotationStyle)
        assert pos_style.font_size == 11.0
        assert pos_style.terminus_font_size == 15.0
        assert pos_style.color == Color("#00FF00")

    def test_parse_style_without_position_annotation(self, tmp_path):
        """Test parsing style file without position annotation section."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[helix]
color = "#FF0000"

[sheet]
color = "#0000FF"
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        # Should not include position_annotation
        assert "position_annotation" not in styles
        assert "helix" in styles
        assert "sheet" in styles

    def test_parse_position_annotation_style_invalid_values(self, tmp_path):
        """Test that invalid values in position annotation style raise validation errors."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = -5.0
text_offset = -1.0
"""
        )

        parser = StyleParser(style_file)

        with pytest.raises(Exception):  # Should raise validation error
            parser.parse()

    def test_parse_position_annotation_style_color_formats(self, tmp_path):
        """Test parsing different color formats for position annotations."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
color = "#FF0000"
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        pos_style = styles["position_annotation"]
        assert pos_style.color == Color("#FF0000")

    def test_get_element_styles_includes_position_annotation(self, tmp_path):
        """Test that get_element_styles method includes position annotations."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = 9.0

[helix]
color = "#FF0000"
"""
        )

        parser = StyleParser(style_file)
        styles = parser.get_element_styles()

        assert "position_annotation" in styles
        assert "helix" in styles
        assert isinstance(styles["position_annotation"], PositionAnnotationStyle)

    def test_style_parser_unknown_section_warning(self, tmp_path, capsys):
        """Test that unknown sections generate warnings but don't fail parsing."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = 8.0

[unknown_section]
some_value = 42

[helix]
color = "#FF0000"
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        # Should parse known sections successfully
        assert "position_annotation" in styles
        assert "helix" in styles
        assert "unknown_section" not in styles

        # Should have printed warning about unknown section
        captured = capsys.readouterr()
        assert "unknown_section" in captured.out.lower()

    def test_position_annotation_style_inheritance(self, tmp_path):
        """Test that position annotation style properly inherits from base style."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
font_size = 9.0
opacity = 0.7
visibility = false
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        pos_style = styles["position_annotation"]

        # Should have position-specific attributes
        assert pos_style.font_size == 9.0

        # Should inherit base style attributes
        assert pos_style.opacity == 0.7
        assert pos_style.visibility is False

    def test_position_annotation_offset_tuples(self, tmp_path):
        """Test parsing tuple values for offset properties."""
        style_file = tmp_path / "style.toml"
        style_file.write_text(
            """
[position_annotation]
offset = [2.0, -3.0]
label_offset = [1.0, 1.5]
"""
        )

        parser = StyleParser(style_file)
        styles = parser.parse()

        pos_style = styles["position_annotation"]
        assert pos_style.offset == (2.0, -3.0)
        assert pos_style.label_offset == (1.0, 1.5)
