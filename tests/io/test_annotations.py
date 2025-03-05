# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for annotation file parsing."""

import os
import tempfile
from pathlib import Path

import pytest
import toml

from flatprot.io.annotations import (
    AnnotationParser,
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    MissingRequiredFieldError,
    InvalidFieldTypeError,
)


@pytest.fixture
def valid_annotations_file():
    """Create a temporary file with valid annotations."""
    content = {
        "annotations": [
            {
                "label": "Catalytic Site",
                "type": "point",
                "indices": 42,
                "chain": "A",
            },
            {
                "label": "Binding Pair",
                "type": "pair",
                "indices": [10, 15],
                "chain": "B",
            },
            {
                "label": "Domain",
                "type": "area",
                "range": {"start": 0, "end": 15},
                "chain": "A",
            },
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def missing_annotations_file():
    """Create a temporary file without annotations list."""
    content = {"not_annotations": []}

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def malformed_toml_file():
    """Create a temporary file with malformed TOML."""
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("This is not valid TOML syntax []\n=")
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


def test_parser_with_nonexistent_file():
    """Test parser with a nonexistent file."""
    with pytest.raises(AnnotationFileNotFoundError):
        AnnotationParser(Path("nonexistent_file.toml"))


def test_parser_with_malformed_toml(malformed_toml_file):
    """Test parser with malformed TOML."""
    parser = AnnotationParser(Path(malformed_toml_file))
    with pytest.raises(MalformedAnnotationError):
        parser.parse()


def test_parser_with_missing_annotations(missing_annotations_file):
    """Test parser with missing annotations list."""
    parser = AnnotationParser(Path(missing_annotations_file))
    with pytest.raises(MalformedAnnotationError):
        parser.parse()


def test_parser_with_valid_file(valid_annotations_file):
    """Test parser with a valid annotations file."""
    parser = AnnotationParser(Path(valid_annotations_file))
    annotations = parser.parse()

    assert len(annotations) == 3

    # Check point annotation
    assert annotations[0]["type"] == "point"
    assert annotations[0]["label"] == "Catalytic Site"
    assert annotations[0]["indices"] == 42
    assert annotations[0]["chain"] == "A"

    # Check pair annotation
    assert annotations[1]["type"] == "pair"
    assert annotations[1]["label"] == "Binding Pair"
    assert annotations[1]["indices"] == [10, 15]
    assert annotations[1]["chain"] == "B"

    # Check area annotation
    assert annotations[2]["type"] == "area"
    assert annotations[2]["label"] == "Domain"
    assert annotations[2]["range"]["start"] == 0
    assert annotations[2]["range"]["end"] == 15
    assert annotations[2]["chain"] == "A"


def test_missing_required_field():
    """Test validation of missing required fields."""
    content = {
        "annotations": [
            {
                "label": "Missing Type",
                # "type" is missing
                "indices": 42,
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file))
        with pytest.raises(MissingRequiredFieldError) as excinfo:
            parser.parse()
        assert "type" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_invalid_field_type():
    """Test validation of invalid field types."""
    content = {
        "annotations": [
            {
                "label": "Invalid Indices Type",
                "type": "point",
                "indices": "not an integer",  # Should be an integer
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file))
        with pytest.raises(InvalidFieldTypeError) as excinfo:
            parser.parse()
        assert "indices" in str(excinfo.value)
        assert "integer" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_invalid_annotation_type():
    """Test validation of invalid annotation type."""
    content = {
        "annotations": [
            {
                "label": "Invalid Type",
                "type": "unknown_type",  # Invalid type
                "indices": 42,
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file))
        with pytest.raises(InvalidFieldTypeError) as excinfo:
            parser.parse()
        assert "type" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_invalid_line_indices():
    """Test validation of invalid line indices."""
    content = {
        "annotations": [
            {
                "label": "Invalid Line Indices",
                "type": "pair",
                "indices": [1, 2, 3],  # Should be exactly 2 indices
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file))
        with pytest.raises(InvalidFieldTypeError) as excinfo:
            parser.parse()
        assert "indices" in str(excinfo.value)
        assert "2 integers" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_invalid_area_range():
    """Test validation of invalid area range."""
    content = {
        "annotations": [
            {
                "label": "Invalid Area Range",
                "type": "area",
                "range": {"start": 15, "end": 5},  # end < start
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file))
        with pytest.raises(InvalidFieldTypeError) as excinfo:
            parser.parse()
        assert "range" in str(excinfo.value)
        assert "greater than or equal to start" in str(excinfo.value)
    finally:
        os.unlink(temp_file)
