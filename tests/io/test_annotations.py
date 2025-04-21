# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for annotation file parsing (src/flatprot/io/annotations.py)."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator, List

import pytest
import toml
from pydantic_extra_types.color import Color

from flatprot.io.annotations import AnnotationParser
from flatprot.io.errors import (
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
)
from flatprot.scene.annotation import (
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
    PointAnnotationStyle,
    LineAnnotationStyle,
    AreaAnnotationStyle,
)
from flatprot.core import ResidueCoordinate, ResidueRange, ResidueRangeSet


# --- Fixtures ---


@pytest.fixture
def valid_annotations_dict() -> Dict[str, List[Dict[str, Any]]]:
    """Provides a dictionary representing a valid TOML annotation structure."""
    return {
        "annotations": [
            {
                "label": "Catalytic Site",
                "type": "point",
                "index": "A:42",  # New format
                "style": {"color": "red", "marker_radius": 1.5},
            },
            {
                "label": "Binding Pair",
                "type": "line",
                "indices": ["B:10", "B:15"],  # New format
                "style": {"color": "#00FF00", "stroke_width": 2.0},
            },
            {
                "label": "Alpha Helix",
                "type": "area",
                "range": "A:100-115",  # New format
                "style": {"color": "blue", "fill_opacity": 0.5},
            },
            {
                "label": "Another Point",  # Annotation without style
                "type": "point",
                "index": "C:1",
            },
        ]
    }


@pytest.fixture
def valid_annotations_file(
    valid_annotations_dict: Dict[str, List[Dict[str, Any]]],
) -> Generator[Path, None, None]:
    """Creates a temporary TOML file with valid annotations."""
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(valid_annotations_dict, f)
        temp_file_path = Path(f.name)
    yield temp_file_path
    os.unlink(temp_file_path)


@pytest.fixture
def missing_annotations_list_file() -> Generator[Path, None, None]:
    """Creates a temporary TOML file missing the top-level 'annotations' list."""
    content = {"some_other_key": []}
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file_path = Path(f.name)
    yield temp_file_path
    os.unlink(temp_file_path)


@pytest.fixture
def annotations_not_a_list_file() -> Generator[Path, None, None]:
    """Creates a temporary TOML file where 'annotations' is not a list."""
    content = {"annotations": {"key": "value"}}  # Should be a list
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file_path = Path(f.name)
    yield temp_file_path
    os.unlink(temp_file_path)


@pytest.fixture
def malformed_toml_file() -> Generator[Path, None, None]:
    """Creates a temporary file with invalid TOML syntax."""
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("this is = not [ valid toml\n")
        temp_file_path = Path(f.name)
    yield temp_file_path
    os.unlink(temp_file_path)


@pytest.fixture
def empty_annotations_list_file() -> Generator[Path, None, None]:
    """Creates a temporary TOML file with an empty 'annotations' list."""
    content = {"annotations": []}
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file_path = Path(f.name)
    yield temp_file_path
    os.unlink(temp_file_path)


# No more mock_scene or mock_scene_element fixtures needed

# --- Test Classes ---


class TestFileLoadingAndStructure:
    """Tests related to loading the file and basic TOML structure validation."""

    def test_parser_with_nonexistent_file(self) -> None:
        """Test parser raises AnnotationFileNotFoundError for nonexistent files."""
        with pytest.raises(AnnotationFileNotFoundError):
            AnnotationParser(Path("nonexistent_file.toml"))

    def test_parser_with_malformed_toml(self, malformed_toml_file: Path) -> None:
        """Test parser raises MalformedAnnotationError for invalid TOML syntax."""
        parser = AnnotationParser(malformed_toml_file)
        with pytest.raises(MalformedAnnotationError, match="Invalid TOML syntax"):
            parser.parse()

    def test_parser_with_missing_annotations_list(
        self, missing_annotations_list_file: Path
    ) -> None:
        """Test parser raises MalformedAnnotationError when 'annotations' list is missing."""
        parser = AnnotationParser(missing_annotations_list_file)
        with pytest.raises(
            MalformedAnnotationError, match="Missing top-level 'annotations' list"
        ):
            parser.parse()

    def test_parser_annotations_not_a_list(
        self, annotations_not_a_list_file: Path
    ) -> None:
        """Test parser raises MalformedAnnotationError if 'annotations' is not a list."""
        parser = AnnotationParser(annotations_not_a_list_file)
        with pytest.raises(
            MalformedAnnotationError, match="'annotations' key must contain a list"
        ):
            parser.parse()

    def test_parse_empty_annotations_list(
        self, empty_annotations_list_file: Path
    ) -> None:
        """Test parsing a file with an empty annotations list returns an empty list."""
        parser = AnnotationParser(empty_annotations_list_file)
        annotations = parser.parse()
        assert annotations == []


class TestAnnotationParsing:
    """Tests parsing valid annotation entries."""

    def test_parse_valid_annotations(self, valid_annotations_file: Path) -> None:
        """Test parsing a file with various valid annotations and styles."""
        parser = AnnotationParser(valid_annotations_file)
        annotations = parser.parse()

        assert len(annotations) == 4

        # --- Check Point Annotation ---
        anno0 = annotations[0]
        assert isinstance(anno0, PointAnnotation)
        assert anno0.label == "Catalytic Site"
        assert anno0.target == ResidueCoordinate("A", 42)
        assert anno0.id.startswith(f"annotation_{valid_annotations_file.stem}_0_point")
        assert isinstance(anno0.style, PointAnnotationStyle)
        assert anno0.style.color == Color("red")
        assert anno0.style.marker_radius == 1.5

        # --- Check Line Annotation ---
        anno1 = annotations[1]
        assert isinstance(anno1, LineAnnotation)
        assert anno1.label == "Binding Pair"
        assert anno1.target == [
            ResidueCoordinate("B", 10),
            ResidueCoordinate("B", 15),
        ]
        assert anno1.id.startswith(f"annotation_{valid_annotations_file.stem}_1_line")
        assert isinstance(anno1.style, LineAnnotationStyle)
        assert anno1.style.color == Color("#00FF00")
        assert anno1.style.stroke_width == 2.0

        # --- Check Area Annotation ---
        anno2 = annotations[2]
        assert isinstance(anno2, AreaAnnotation)
        assert anno2.label == "Alpha Helix"
        assert anno2.target == ResidueRangeSet([ResidueRange("A", 100, 115)])
        assert anno2.id.startswith(f"annotation_{valid_annotations_file.stem}_2_area")
        assert isinstance(anno2.style, AreaAnnotationStyle)
        assert anno2.style.color == Color("blue")
        assert anno2.style.fill_opacity == 0.5

        # --- Check Point Annotation (No Style) ---
        anno3 = annotations[3]
        assert isinstance(anno3, PointAnnotation)
        assert anno3.label == "Another Point"
        assert anno3.target == ResidueCoordinate("C", 1)
        assert anno3.id.startswith(f"annotation_{valid_annotations_file.stem}_3_point")
        assert (
            isinstance(anno3.style, PointAnnotationStyle)
            and anno3.style == anno3.default_style
        )

    def test_parse_annotation_without_label(self) -> None:
        """Test parsing an annotation where the optional 'label' is missing."""
        content = {
            "annotations": [
                {
                    # No label field
                    "type": "point",
                    "index": "A:10",
                }
            ]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            annotations = parser.parse()
            assert len(annotations) == 1
            assert isinstance(annotations[0], PointAnnotation)
            assert annotations[0].label is None  # Label should be None
            assert annotations[0].target == ResidueCoordinate("A", 10)
        finally:
            os.unlink(temp_file_path)


class TestAnnotationFormatValidation:
    """Tests for validating the format of individual annotation entries."""

    @pytest.mark.parametrize(
        "entry, expected_error_msg",
        [
            ("not a dictionary", "Annotation entry must be a table"),
            (
                {"type": "point"},
                "Missing 'index' field for 'point' annotation",
            ),  # Missing index
            (
                {"type": "line"},
                "Field 'indices' for 'line' annotation must be a list",
            ),  # Missing indices
            (
                {"type": "area"},
                "Missing 'range' field for 'area' annotation",
            ),  # Missing range
            ({"index": "A:1"}, "Missing or invalid 'type' field"),  # Missing type
            (
                {"type": 123, "index": "A:1"},
                "Missing or invalid 'type' field",
            ),  # Invalid type (not str)
            (
                {"type": "unknown", "index": "A:1"},
                "Unknown annotation 'type': 'unknown'",
            ),  # Unknown type
        ],
        ids=[
            "entry_not_dict",
            "point_missing_index",
            "line_missing_indices",
            "area_missing_range",
            "missing_type",
            "invalid_type_non_str",
            "unknown_type",
        ],
    )
    def test_missing_or_invalid_required_fields(
        self, entry: Any, expected_error_msg: str
    ) -> None:
        """Test parser raises MalformedAnnotationError for missing/invalid required fields."""
        content = {"annotations": [entry]}
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            # Use a simple dumper if entry is not a dict for testing that case
            if isinstance(entry, dict):
                toml.dump(content, f)
            else:
                # Write non-dict entry manually (ensure valid TOML for the list itself)
                f.write(f"annotations = [ {repr(entry)} ]\n")
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert expected_error_msg in str(excinfo.value)
        finally:
            os.unlink(temp_file_path)

    # --- Point Specific Format Tests ---
    @pytest.mark.parametrize(
        "index_value, expected_error_msg",
        [
            (123, "Expected string for coordinate, got int"),  # Wrong type
            ("A123", "Invalid coordinate format 'A123'"),  # Missing colon
            ("A:", "Invalid coordinate format 'A:'"),  # Missing index
            (":123", "Invalid coordinate format ':123'"),  # Missing chain
            ("A:12:34", "Invalid coordinate format 'A:12:34'"),  # Too many colons
            ("A:abc", "Invalid coordinate format 'A:abc'"),  # Non-numeric index
        ],
        ids=[
            "int_type",
            "no_colon",
            "missing_index",
            "missing_chain",
            "too_many_colons",
            "non_numeric_index",
        ],
    )
    def test_invalid_point_index_format(
        self, index_value: Any, expected_error_msg: str
    ) -> None:
        """Test invalid formats for the 'index' field of point annotations."""
        content = {
            "annotations": [{"type": "point", "label": "Test", "index": index_value}]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert expected_error_msg in str(excinfo.value)
        finally:
            os.unlink(temp_file_path)

    # --- Line Specific Format Tests ---
    @pytest.mark.parametrize(
        "indices_value, expected_error_msg",
        [
            (
                "A:10, A:20",
                "Field 'indices' for 'line' annotation must be a list",
            ),  # String, not list
            (
                ["A:10"],
                "must be a list of exactly two coordinate strings",
            ),  # List too short
            (
                ["A:10", "B:20", "C:30"],
                "must be a list of exactly two coordinate strings",
            ),  # List too long
            (
                ["A:10", "B:abc"],
                "Invalid coordinate format 'B:abc'",
            ),  # Invalid format in list
        ],
        ids=[
            "string_not_list",
            "list_too_short",
            "list_too_long",
            "invalid_format_in_list",
        ],
    )
    def test_invalid_line_indices_format(
        self, indices_value: Any, expected_error_msg: str
    ) -> None:
        """Test invalid formats for the 'indices' field of line annotations."""
        content = {
            "annotations": [{"type": "line", "label": "Test", "indices": indices_value}]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert expected_error_msg in str(excinfo.value)
        finally:
            os.unlink(temp_file_path)

    # --- Area Specific Format Tests ---
    @pytest.mark.parametrize(
        "range_value, expected_error_msg",
        [
            (100, "Expected string for range, got int"),  # Wrong type
            ("A100-150", "Invalid range format 'A100-150'"),  # Missing colon
            ("A:100", "Invalid range format 'A:100'"),  # Missing dash and end
            ("A:100-", "Invalid range format 'A:100-'"),  # Missing end
            ("A:-150", "Invalid range format 'A:-150'"),  # Missing start
            (
                "A:100-150-200",
                "Invalid range format 'A:100-150-200'",
            ),  # Too many dashes
            ("A:100-abc", "Invalid range format 'A:100-abc'"),  # Non-numeric end
            ("A:abc-150", "Invalid range format 'A:abc-150'"),  # Non-numeric start
            (
                "A:150-100",
                "Start index (150) cannot be greater than end index (100)",
            ),  # Start > End
        ],
        ids=[
            "int_type",
            "no_colon",
            "missing_dash_end",
            "missing_end",
            "missing_start",
            "too_many_dashes",
            "non_numeric_end",
            "non_numeric_start",
            "start_greater_than_end",
        ],
    )
    def test_invalid_area_range_format(
        self, range_value: Any, expected_error_msg: str
    ) -> None:
        """Test invalid formats for the 'range' field of area annotations."""
        content = {
            "annotations": [{"type": "area", "label": "Test", "range": range_value}]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert expected_error_msg in str(excinfo.value)
        finally:
            os.unlink(temp_file_path)


class TestInlineStyleParsing:
    """Tests parsing of the optional 'style' field."""

    def test_parse_valid_inline_styles(self, valid_annotations_file: Path) -> None:
        """Test that valid style dictionaries are correctly parsed into Pydantic models."""
        # This is implicitly tested in TestAnnotationParsing.test_parse_valid_annotations
        # We re-assert the style types and some values here for clarity.
        parser = AnnotationParser(valid_annotations_file)
        annotations = parser.parse()

        assert isinstance(annotations[0].style, PointAnnotationStyle)
        assert annotations[0].style.color == Color("red")

        assert isinstance(annotations[1].style, LineAnnotationStyle)
        assert annotations[1].style.stroke_width == 2.0

        assert isinstance(annotations[2].style, AreaAnnotationStyle)
        assert annotations[2].style.fill_opacity == 0.5

        assert (
            isinstance(annotations[3].style, PointAnnotationStyle)
            and annotations[3].style == annotations[3].default_style
        )

    def test_invalid_style_type(self) -> None:
        """Test parsing fails if 'style' is not a dictionary (table)."""
        content = {
            "annotations": [
                {
                    "type": "point",
                    "index": "A:1",
                    "style": "not a dictionary",  # Invalid style type
                }
            ]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert "'style' entry must be a table" in str(excinfo.value)
            assert "got str" in str(excinfo.value)
        finally:
            os.unlink(temp_file_path)

    def test_invalid_style_content_value_type(self) -> None:
        """Test parsing fails if a style field has the wrong value type."""
        content = {
            "annotations": [
                {
                    "type": "point",
                    "index": "A:1",
                    "style": {
                        "color": "red",
                        "marker_radius": "should be float",  # Incorrect type
                    },
                }
            ]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)
        try:
            parser = AnnotationParser(temp_file_path)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            # Pydantic v2 error message format:
            assert "Invalid style definition:" in str(excinfo.value)
            assert "marker_radius:" in str(excinfo.value)  # Field name
            assert "Input should be a valid number" in str(
                excinfo.value
            )  # Pydantic error
        finally:
            os.unlink(temp_file_path)

    def test_invalid_style_content_unknown_field(self) -> None:
        """Test parsing fails if style contains an unknown field (strict parsing)."""
        content = {
            "annotations": [
                {
                    "type": "point",
                    "index": "A:1",
                    "style": {
                        "color": "red",
                        "unknown_style_field": True,  # Extra field
                    },
                }
            ]
        }
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file_path = Path(f.name)

        with pytest.raises(MalformedAnnotationError):
            parser = AnnotationParser(temp_file_path)
            _ = parser.parse()

        os.unlink(temp_file_path)
