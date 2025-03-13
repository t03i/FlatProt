# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for annotation file parsing."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import toml
import numpy as np

from flatprot.io.annotations import (
    AnnotationParser,
)
from flatprot.io.errors import (
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    InvalidReferenceError,
)
from flatprot.scene.annotations.point import PointAnnotation
from flatprot.scene.annotations.line import LineAnnotation
from flatprot.scene.annotations.area import AreaAnnotation
from flatprot.scene.elements import SceneElement
from flatprot.scene import Scene


@pytest.fixture
def valid_annotations_file():
    """Create a temporary file with valid annotations."""
    content = {
        "annotations": [
            {
                "label": "Catalytic Site",
                "type": "point",
                "index": 42,
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


@pytest.fixture
def mock_scene_element():
    """Create a mock scene element."""
    element = MagicMock(spec=SceneElement)
    element.display_coordinates = np.array([0.0, 0.0])
    element.calculate_display_coordinates_at_resiude = lambda residue_idx: np.array(
        [float(residue_idx), float(residue_idx)]
    )
    return element


@pytest.fixture
def mock_scene(mock_scene_element):
    """Create a mock scene object."""
    scene = MagicMock(spec=Scene)

    # Set up mock get_elements_for_residue
    def mock_get_elements_for_residue(chain_id, residue_idx):
        if chain_id == "A" and 0 <= residue_idx <= 100 and residue_idx != 5:
            return [mock_scene_element]
        elif chain_id == "B" and 0 <= residue_idx <= 50:
            return [mock_scene_element]
        else:
            return []

    scene.get_elements_for_residue.side_effect = mock_get_elements_for_residue

    # Set up mock get_elements_for_residue_range
    def mock_get_elements_for_residue_range(chain_id, start, end):
        if start > end:
            return []

        elements = []
        if chain_id == "A":
            for i in range(start, min(end + 1, 101)):
                if i != 5:  # Skip residue 5 to simulate discontinuity
                    elements.append(mock_scene_element)
        elif chain_id == "B":
            for i in range(start, min(end + 1, 51)):
                elements.append(mock_scene_element)

        return elements

    scene.get_elements_for_residue_range.side_effect = (
        mock_get_elements_for_residue_range
    )

    return scene


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


def test_parser_with_valid_file_no_scene(valid_annotations_file):
    """Test parser with a valid annotations file but no scene."""
    parser = AnnotationParser(Path(valid_annotations_file))
    annotations = parser.parse()

    # Without a scene, we just validate the TOML but don't create objects
    assert len(annotations) == 0


def test_parser_with_scene(valid_annotations_file, mock_scene):
    """Test parser with a valid annotations file and scene."""
    parser = AnnotationParser(Path(valid_annotations_file), scene=mock_scene)
    annotations = parser.parse()

    assert len(annotations) == 3

    # Check point annotation
    assert isinstance(annotations[0], PointAnnotation)
    assert annotations[0].label == "Catalytic Site"
    assert len(annotations[0].targets) == 1
    assert annotations[0].indices == [42]

    # Check pair annotation
    assert isinstance(annotations[1], LineAnnotation)
    assert annotations[1].label == "Binding Pair"
    assert len(annotations[1].targets) == 2
    assert annotations[1].indices == [10, 15]

    # Check area annotation
    assert isinstance(annotations[2], AreaAnnotation)
    assert annotations[2].label == "Domain"
    assert len(annotations[2].targets) > 0
    assert annotations[2].indices is None


def test_invalid_point_type():
    """Test validation of invalid point type."""
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
        with pytest.raises(MalformedAnnotationError) as excinfo:
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
        with pytest.raises(MalformedAnnotationError) as excinfo:
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
        with pytest.raises(MalformedAnnotationError) as excinfo:
            parser.parse()
        assert "indices" in str(excinfo.value)
        assert "2 indices" in str(excinfo.value)
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
        with pytest.raises(MalformedAnnotationError) as excinfo:
            parser.parse()
        assert "greater than or equal to start" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_missing_required_field():
    """Test validation of missing required fields."""
    content = {
        "annotations": [
            {
                "label": "Missing Chain",
                "type": "point",
                "indices": 42,
                # "chain" is missing
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file))
        with pytest.raises(MalformedAnnotationError) as excinfo:
            parser.parse()
        assert "chain" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_invalid_residue_reference(mock_scene):
    """Test validation of invalid residue reference."""
    content = {
        "annotations": [
            {
                "label": "Invalid Residue",
                "type": "point",
                "index": 200,  # Residue 200 doesn't exist in chain A
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        # Our mock_scene only has residues 0-100 for chain A
        parser = AnnotationParser(Path(temp_file), scene=mock_scene)
        with pytest.raises(InvalidReferenceError) as excinfo:
            parser.parse()
        assert "residue" in str(excinfo.value)
        assert "200" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_invalid_chain_reference(mock_scene):
    """Test validation of invalid chain reference."""
    content = {
        "annotations": [
            {
                "label": "Invalid Chain",
                "type": "point",
                "index": 42,
                "chain": "C",  # Chain C doesn't exist
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        # Our mock_scene only has chains A and B
        parser = AnnotationParser(Path(temp_file), scene=mock_scene)
        with pytest.raises(InvalidReferenceError) as excinfo:
            parser.parse()
        assert "residue" in str(excinfo.value)
        assert "chain C" in str(excinfo.value)
    finally:
        os.unlink(temp_file)


def test_discontinuous_range_annotation(mock_scene):
    """Test annotation for a range with discontinuities."""
    content = {
        "annotations": [
            {
                "label": "Discontinuous Area",
                "type": "area",
                "range": {"start": 0, "end": 10},  # Includes missing residue 5
                "chain": "A",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    try:
        parser = AnnotationParser(Path(temp_file), scene=mock_scene)
        annotations = parser.parse()

        # Check that the annotation was created correctly
        assert len(annotations) == 1
        assert isinstance(annotations[0], AreaAnnotation)
        assert annotations[0].label == "Discontinuous Area"

        # Should have 10 targets (0-10 range minus the missing residue 5)
        assert len(annotations[0].targets) == 10
    finally:
        os.unlink(temp_file)
