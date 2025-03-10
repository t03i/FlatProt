# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for style parser"""

import os
import tempfile
import pytest
from pathlib import Path

from flatprot.io.styles import StyleParser, InvalidTomlError, StyleValidationError
from flatprot.style.types import StyleType
from flatprot.style.structure import HelixStyle, SheetStyle, ElementStyle
from flatprot.style.annotation import AreaAnnotationStyle


def create_temp_toml(content: str) -> Path:
    """Create a temporary TOML file with the given content."""
    fd, path = tempfile.mkstemp(suffix=".toml")
    os.write(fd, content.encode())
    os.close(fd)
    return Path(path)


def test_valid_style_file():
    """Test parsing a valid style file."""
    content = """
    [helix]
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = 5.0

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point]
    fill_color = "#FFC300"
    radius = 3.5

    [line]
    stroke_color = "#900C3F"
    stroke_width = 1.5

    [area]
    fill_color = "#C70039"
    fill_opacity = 0.7
    stroke_color = "#900C3F"
    padding = 5.0
    smoothing_window = 3
    interpolation_points = 100
    """

    file_path = create_temp_toml(content)
    try:
        parser = StyleParser(file_path)
        style_data = parser.get_style_data()

        # Test that all sections are present
        assert "helix" in style_data
        assert "sheet" in style_data
        assert "point" in style_data
        assert "line" in style_data
        assert "area" in style_data

        # Test specific values
        assert style_data["helix"]["fill_color"] == "#FF5733"
        assert style_data["sheet"]["line_width"] == 2.5
        assert style_data["point"]["radius"] == 3.5
        assert style_data["line"]["stroke_width"] == 1.5
        assert style_data["area"]["interpolation_points"] == 100
    finally:
        os.unlink(file_path)


def test_missing_section():
    """Test validation for missing sections."""
    # Missing 'area' section
    content = """
    [helix]
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = 5.0

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point]
    fill_color = "#FFC300"
    radius = 3.5

    [line]
    stroke_color = "#900C3F"
    stroke_width = 1.5
    """

    file_path = create_temp_toml(content)
    try:
        with pytest.raises(StyleValidationError) as excinfo:
            StyleParser(file_path)
        assert "Missing required style sections: area" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_malformed_toml():
    """Test validation for malformed TOML."""
    # Malformed TOML (missing closing bracket)
    content = """
    [helix
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = 5.0
    """

    file_path = create_temp_toml(content)
    try:
        with pytest.raises(InvalidTomlError) as excinfo:
            StyleParser(file_path)
        assert "Invalid TOML format" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError) as excinfo:
        StyleParser("non_existent_file.toml")
    assert "Style file not found" in str(excinfo.value)


def test_create_style_objects():
    """Test creation of style objects from the parsed data."""
    content = """
    [helix]
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = 5.0

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point]
    fill_color = "#FFC300"
    radius = 3.5

    [line]
    stroke_color = "#900C3F"
    stroke_width = 1.5

    [area]
    fill_color = "#C70039"
    fill_opacity = 0.7
    stroke_color = "#900C3F"
    padding = 5.0
    smoothing_window = 3
    interpolation_points = 100
    """

    file_path = create_temp_toml(content)
    try:
        parser = StyleParser(file_path)

        # Test getting all styles
        styles = parser.get_styles()
        assert StyleType.HELIX in styles
        assert StyleType.SHEET in styles
        assert StyleType.POINT in styles
        assert StyleType.ELEMENT in styles
        assert StyleType.AREA_ANNOTATION in styles

        # Test individual style objects
        helix_style = parser.create_style("helix")
        assert isinstance(helix_style, HelixStyle)
        assert helix_style.fill_color.as_hex() == "#ff5733"
        assert helix_style.amplitude == 5.0

        sheet_style = parser.create_style("sheet")
        assert isinstance(sheet_style, SheetStyle)
        assert sheet_style.fill_color.as_hex() == "#daf7a6"
        assert sheet_style.line_width == 2.5
        assert sheet_style.min_sheet_length == 3

        point_style = parser.create_style("point")
        assert isinstance(point_style, ElementStyle)
        assert point_style.fill_color.as_hex() == "#ffc300"
        assert point_style.line_width == 7.0  # radius * 2

        line_style = parser.create_style("line")
        assert isinstance(line_style, ElementStyle)
        assert line_style.stroke_color.as_hex() == "#900c3f"
        assert line_style.line_width == 1.5

        area_style = parser.create_style("area")
        assert isinstance(area_style, AreaAnnotationStyle)
        assert area_style.fill_color.as_hex() == "#c70039"
        assert area_style.fill_opacity == 0.7
        assert area_style.stroke_color.as_hex() == "#900c3f"
        assert area_style.padding == 5.0
        assert area_style.smoothing_window == 3
        assert area_style.interpolation_points == 100
    finally:
        os.unlink(file_path)


def test_invalid_color_format():
    """Test validation of invalid color formats."""
    content = """
    [helix]
    fill_color = "not-a-color"
    stroke_color = "#900C3F"
    amplitude = 5.0

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point]
    fill_color = "#FFC300"
    radius = 3.5

    [line]
    stroke_color = "#900C3F"
    stroke_width = 1.5

    [area]
    fill_color = "#C70039"
    fill_opacity = 0.7
    stroke_color = "#900C3F"
    padding = 5.0
    smoothing_window = 3
    interpolation_points = 100
    """

    file_path = create_temp_toml(content)
    try:
        parser = StyleParser(file_path)
        with pytest.raises(StyleValidationError) as excinfo:
            parser.create_style("helix")
        assert "Invalid color format" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_invalid_field_type():
    """Test validation for invalid field types."""
    content = """
    [helix]
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = "not a number"  # Should be a float

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point]
    fill_color = "#FFC300"
    radius = 3.5

    [line]
    stroke_color = "#900C3F"
    stroke_width = 1.5

    [area]
    fill_color = "#C70039"
    fill_opacity = 0.7
    stroke_color = "#900C3F"
    padding = 5.0
    smoothing_window = 3
    interpolation_points = 100
    """

    file_path = create_temp_toml(content)
    try:
        parser = StyleParser(file_path)
        with pytest.raises(StyleValidationError) as excinfo:
            parser.create_style("helix")
        assert "helix" in str(excinfo.value)
    finally:
        os.unlink(file_path)


def test_invalid_value_range():
    """Test validation for invalid value ranges."""
    content = """
    [helix]
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = -5.0  # Should be positive

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point]
    fill_color = "#FFC300"
    radius = 3.5

    [line]
    stroke_color = "#900C3F"
    stroke_width = 1.5

    [area]
    fill_color = "#C70039"
    fill_opacity = 0.7
    stroke_color = "#900C3F"
    padding = 5.0
    smoothing_window = 3
    interpolation_points = 100
    """

    file_path = create_temp_toml(content)
    try:
        parser = StyleParser(file_path)
        with pytest.raises(StyleValidationError) as excinfo:
            parser.create_style("helix")
        assert "helix" in str(excinfo.value)
    finally:
        os.unlink(file_path)
