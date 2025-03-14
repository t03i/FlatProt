# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for style parser"""

import os
import tempfile
import pytest
from pathlib import Path

from flatprot.io.styles import StyleParser


from flatprot.style.types import StyleType
from flatprot.style.structure import HelixStyle, SheetStyle
from flatprot.style.annotation import AreaAnnotationStyle
from flatprot.io import (
    InvalidTomlError,
    StyleValidationError,
    StyleParsingError,
    StyleFileNotFoundError,
)


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
    with pytest.raises(StyleFileNotFoundError) as excinfo:
        StyleParser("non_existent_file.toml")
    assert "Style file not found" in str(excinfo.value)


def test_style_manager_creation():
    """Test creation of style manager from the parsed data."""
    content = """
    [helix]
    fill_color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = 5.0

    [sheet]
    fill_color = "#DAF7A6"
    line_width = 2.5
    min_sheet_length = 3

    [point_annotation]
    fill_color = "#FFC300"
    radius = 3.5

    [line_annotation]
    stroke_color = "#900C3F"
    stroke_width = 1.5

    [area_annotation]
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

        # Test getting style manager
        style_manager = parser.get_style_manager()

        # Test that style manager has the correct styles
        helix_style = style_manager.get_style(StyleType.HELIX)
        assert isinstance(helix_style, HelixStyle)
        assert helix_style.fill_color.as_hex() == "#ff5733"
        assert helix_style.amplitude == 5.0

        sheet_style = style_manager.get_style(StyleType.SHEET)
        assert isinstance(sheet_style, SheetStyle)
        assert sheet_style.fill_color.as_hex() == "#daf7a6"
        assert sheet_style.line_width == 2.5
        assert sheet_style.min_sheet_length == 3

        # Test special handling for point style (radius conversion)
        point_style = style_manager.get_style(StyleType.POINT_ANNOTATION)
        assert point_style.stroke_width == 7.0  # radius * 2

        # Test special handling for line style (stroke_width conversion)
        line_style = style_manager.get_style(StyleType.LINE_ANNOTATION)
        assert line_style.stroke_color.as_hex() == "#900c3f"
        assert line_style.stroke_width == 1.5

        # Test for area
        area_style = style_manager.get_style(StyleType.AREA_ANNOTATION)
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
        with pytest.raises(StyleParsingError) as excinfo:
            parser.get_style_manager()
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
            parser.get_style_manager()
        assert "Invalid style definition" in str(excinfo.value)
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
            parser.get_style_manager()
        assert "Invalid style definition" in str(excinfo.value)
    finally:
        os.unlink(file_path)
