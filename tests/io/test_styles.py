# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for src/flatprot/io/styles.py"""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic_extra_types.color import Color

from flatprot.io.styles import StyleParser

# Import ConnectionStyle for testing
from flatprot.scene.connection import ConnectionStyle
from flatprot.scene.structure import (
    HelixStyle,
    SheetStyle,
)
from flatprot.io.errors import (
    InvalidTomlError,
    StyleValidationError,
    StyleFileNotFoundError,
)


def create_temp_toml(content: str) -> Path:
    """Helper function to create a temporary TOML file with the given content."""
    # Use delete=False and manual unlinking in a finally block for more robust cleanup
    # across different OS and test runners.
    f = tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


# --- Test Fixtures ---


@pytest.fixture
def valid_style_content() -> str:
    """Provides valid TOML content for structure styles."""
    return """
    [helix]
    color = "#FF5733"
    stroke_color = "#900C3F"
    amplitude = 5.0
    wavelength = 0.5
    stroke_width = 3.0

    [sheet]
    color = "#daf7a6"
    stroke_color = "black"
    stroke_width = 2.5
    min_sheet_length = 3

    [connection]
    color = "gray"
    stroke_width = 1.5
    line_style = "dashed"

    [unknown_section] # This should be ignored with a warning
    some_value = 123
    """


@pytest.fixture
def valid_style_file(valid_style_content: str) -> Path:
    """Creates a temporary file with valid structure style TOML content."""
    file_path = create_temp_toml(valid_style_content)
    yield file_path
    os.unlink(file_path)  # Cleanup


@pytest.fixture
def partial_style_content() -> str:
    """Provides TOML content with only the helix style defined."""
    return """
    [helix]
    color = "red"
    amplitude = 4
    """


@pytest.fixture
def partial_style_file(partial_style_content: str) -> Path:
    """Creates a temporary file with partial structure style TOML content."""
    file_path = create_temp_toml(partial_style_content)
    yield file_path
    os.unlink(file_path)  # Cleanup


# --- Test Cases ---


class TestStyleParserInitialization:
    """Tests for StyleParser initialization and basic file handling."""

    def test_init_valid_file(self, valid_style_file: Path, capsys) -> None:
        """Test successful initialization with a valid TOML file."""
        try:
            parser = StyleParser(valid_style_file)
            capsys.readouterr()
            assert parser.file_path == valid_style_file
            # Check if raw data is loaded (basic check)
            assert "helix" in parser.get_raw_data()
            assert "sheet" in parser.get_raw_data()
            assert "connection" in parser.get_raw_data()
            assert "unknown_section" in parser.get_raw_data()
        except Exception as e:
            pytest.fail(f"StyleParser initialization failed unexpectedly: {e}")

    def test_init_malformed_toml(self) -> None:
        """Test initialization raises InvalidTomlError for malformed TOML."""
        content = '[helix\ncolor = "red"'  # Malformed TOML
        file_path = create_temp_toml(content)
        try:
            with pytest.raises(InvalidTomlError, match="Invalid TOML format"):
                StyleParser(file_path)
        finally:
            os.unlink(file_path)

    def test_init_file_not_found(self) -> None:
        """Test initialization raises StyleFileNotFoundError for non-existent files."""
        with pytest.raises(StyleFileNotFoundError, match="Style file not found"):
            StyleParser(Path("non_existent_style_file.toml"))

    def test_init_valid_file_with_unknown_section(
        self, capsys, valid_style_file: Path
    ) -> None:
        """Test initialization with unknown sections logs a warning."""
        # Initialization happens here, which triggers _validate_structure
        StyleParser(valid_style_file)
        captured = capsys.readouterr()
        assert (
            "Warning: Unknown style sections found and ignored: unknown_section"
            in captured.out
        )


class TestStyleParsing:
    """Tests for parsing style data into Pydantic models."""

    def test_parse_valid_styles(self, valid_style_file: Path, capsys) -> None:
        """Test parsing a valid style file returns correct Pydantic objects."""
        parser = StyleParser(valid_style_file)
        capsys.readouterr()
        parsed_styles = parser.parse()

        assert isinstance(parsed_styles, dict)
        assert "helix" in parsed_styles
        assert "sheet" in parsed_styles
        assert "coil" not in parsed_styles  # Coil section was missing in the file
        assert "connection" in parsed_styles
        assert "unknown_section" not in parsed_styles  # Unknown sections are not parsed

        # Check HelixStyle
        helix_style = parsed_styles["helix"]
        assert isinstance(helix_style, HelixStyle)
        assert helix_style.color.as_hex() == Color("#ff5733").as_hex()
        assert helix_style.stroke_color.as_hex() == Color("#900c3f").as_hex()
        assert helix_style.amplitude == 5.0
        assert helix_style.wavelength == 0.5
        assert helix_style.stroke_width == 3.0

        # Check SheetStyle
        sheet_style = parsed_styles["sheet"]
        assert isinstance(sheet_style, SheetStyle)
        assert sheet_style.color.as_hex() == Color("#daf7a6").as_hex()
        assert sheet_style.stroke_color.as_hex() == Color("#000000").as_hex()  # 'black'
        assert sheet_style.stroke_width == 2.5
        assert sheet_style.min_sheet_length == 3

        # Check ConnectionStyle
        conn_style = parsed_styles["connection"]
        assert isinstance(conn_style, ConnectionStyle)
        assert conn_style.color.as_hex() == Color("#808080").as_hex()
        assert conn_style.stroke_width == 1.5
        assert conn_style.line_style == "dashed"

    def test_parse_partial_styles(self, partial_style_file: Path) -> None:
        """Test parsing a file with only a subset of known sections."""
        parser = StyleParser(partial_style_file)
        parsed_styles = parser.parse()

        assert list(parsed_styles.keys()) == ["helix"]
        assert isinstance(parsed_styles["helix"], HelixStyle)
        assert (
            parsed_styles["helix"].color.as_hex() == Color("#ff0000").as_hex()
        )  # 'red'
        assert parsed_styles["helix"].amplitude == 4.0

    def test_parse_empty_file(self) -> None:
        """Test parsing an empty TOML file returns an empty dict."""
        content = ""  # Empty content
        file_path = create_temp_toml(content)
        try:
            parser = StyleParser(file_path)
            parsed_styles = parser.parse()
            assert parsed_styles == {}
        finally:
            os.unlink(file_path)

    def test_parse_only_unknown_sections(self, capsys) -> None:
        """Test parsing a file with only unknown sections returns an empty dict."""
        content = """
        [unknown1]
        value = 1
        [unknown2]
        name = "test"
        """
        file_path = create_temp_toml(content)
        try:
            parser = StyleParser(file_path)
            parsed_styles = parser.parse()
            assert parsed_styles == {}
            captured = capsys.readouterr()
            assert (
                "Warning: Unknown style sections found and ignored: unknown1, unknown2"
                in captured.out
            )
        finally:
            os.unlink(file_path)

    def test_section_not_a_table(self) -> None:
        """Test parsing raises StyleValidationError if a known section is not a table."""
        content = """
        helix = "this should be a table"
        """
        file_path = create_temp_toml(content)
        try:
            parser = StyleParser(file_path)
            with pytest.raises(StyleValidationError, match="Expected a table.*got str"):
                parser.parse()
        finally:
            os.unlink(file_path)


class TestStyleValidationErrors:
    """Tests for validation errors during parsing (StyleValidationError)."""

    def test_invalid_color_format(self) -> None:
        """Test StyleValidationError for invalid color formats."""
        content = """
        [sheet]
        color = "not-a-valid-color"
        stroke_width = 2.0
        """
        file_path = create_temp_toml(content)
        parser = StyleParser(file_path)
        with pytest.raises(StyleValidationError) as excinfo:
            parser.parse()
        assert "Invalid style definition in section 'sheet'" in str(excinfo.value)
        assert "color:" in str(excinfo.value)
        os.unlink(file_path)

    def test_invalid_field_type(self) -> None:
        """Test StyleValidationError for invalid field types."""
        content = """
        [helix]
        color = "blue"
        amplitude = "should be a float"
        """
        file_path = create_temp_toml(content)
        try:
            parser = StyleParser(file_path)
            with pytest.raises(StyleValidationError) as excinfo:
                parser.parse()
            assert "Invalid style definition in section 'helix'" in str(excinfo.value)
            assert "amplitude:" in str(excinfo.value)
            # Pydantic v2 error message
            assert "Input should be a valid number" in str(excinfo.value)
        finally:
            os.unlink(file_path)

    def test_invalid_connection_line_style(self) -> None:
        """Test StyleValidationError for invalid connection line_style."""
        content = """
        [connection]
        color = "green"
        line_style = "dotted-dash-what?" # Invalid line style
        """
        file_path = create_temp_toml(content)
        try:
            parser = StyleParser(file_path)
            with pytest.raises(StyleValidationError) as excinfo:
                parser.parse()
            assert "Invalid style definition in section 'connection'" in str(
                excinfo.value
            )
            assert "line_style:" in str(excinfo.value)
            # Check for part of the Pydantic error message related to literals
            assert "Input should be 'solid', 'dashed' or 'dotted'" in str(excinfo.value)
        finally:
            os.unlink(file_path)


class TestGetElementStyles:
    """Tests for the get_element_styles() method."""

    def test_get_element_styles_success(self, valid_style_file: Path, capsys) -> None:
        """Test get_element_styles returns the same as parse on success."""
        parser = StyleParser(valid_style_file)
        capsys.readouterr()
        parsed_direct = parser.parse()
        parsed_via_getter = parser.get_element_styles()
        assert parsed_direct == parsed_via_getter
        assert "helix" in parsed_via_getter
        assert "sheet" in parsed_via_getter
        assert "connection" in parsed_via_getter


class TestGetRawData:
    """Tests for the get_raw_data() method."""

    def test_get_raw_data(
        self, valid_style_content: str, valid_style_file: Path, capsys
    ) -> None:
        """Test get_raw_data returns the original loaded dictionary."""
        import toml  # Import here for comparison

        parser = StyleParser(valid_style_file)
        capsys.readouterr()
        raw_data = parser.get_raw_data()
        expected_raw_data = toml.loads(valid_style_content)
        assert raw_data == expected_raw_data
