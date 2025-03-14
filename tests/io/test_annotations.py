# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for annotation file parsing."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator, List

import pytest
import toml
import numpy as np
from pytest_mock import MockFixture

from flatprot.io.annotations import AnnotationParser
from flatprot.io.errors import (
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    InvalidReferenceError,
    AnnotationError,
)
from flatprot.scene.annotations.point import PointAnnotation
from flatprot.scene.annotations.line import LineAnnotation
from flatprot.scene.annotations.area import AreaAnnotation
from flatprot.scene.elements import SceneElement
from flatprot.scene import Scene


@pytest.fixture
def valid_annotations_dict() -> Dict[str, List[Dict[str, Any]]]:
    """Create a dictionary with valid annotation data.

    Returns:
        Dict containing valid annotations of different types.
    """
    return {
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


@pytest.fixture
def valid_annotations_file(
    valid_annotations_dict: Dict[str, List[Dict[str, Any]]],
) -> Generator[str, None, None]:
    """Create a temporary file with valid annotations.

    Args:
        valid_annotations_dict: Dictionary with valid annotation data.

    Yields:
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(valid_annotations_dict, f)
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def missing_annotations_file() -> Generator[str, None, None]:
    """Create a temporary file without annotations list.

    Yields:
        Path to the temporary file.
    """
    content = {"not_annotations": []}

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        toml.dump(content, f)
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def malformed_toml_file() -> Generator[str, None, None]:
    """Create a temporary file with malformed TOML.

    Yields:
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("This is not valid TOML syntax []\n=")
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def mock_scene_element(mocker: MockFixture) -> Any:
    """Create a mock scene element.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        Mock scene element.
    """
    element = mocker.MagicMock(spec=SceneElement)
    element.display_coordinates = np.array([0.0, 0.0])
    element.calculate_display_coordinates_at_resiude = lambda residue_idx: np.array(
        [float(residue_idx), float(residue_idx)]
    )
    return element


@pytest.fixture
def mock_scene(mocker: MockFixture, mock_scene_element: Any) -> Any:
    """Create a mock scene object.

    Args:
        mocker: Pytest-mock fixture.
        mock_scene_element: Mock scene element fixture.

    Returns:
        Mock scene object.
    """
    scene = mocker.MagicMock(spec=Scene)

    # Set up mock get_elements_for_residue
    def mock_get_elements_for_residue(chain_id: str, residue_idx: int) -> List[Any]:
        if chain_id == "A" and 0 <= residue_idx <= 100 and residue_idx != 5:
            return [mock_scene_element]
        elif chain_id == "B" and 0 <= residue_idx <= 50:
            return [mock_scene_element]
        else:
            return []

    scene.get_elements_for_residue.side_effect = mock_get_elements_for_residue

    # Set up mock get_elements_for_residue_range
    def mock_get_elements_for_residue_range(
        chain_id: str, start: int, end: int
    ) -> List[Any]:
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

    # Set up mock get_element_index_from_global_index
    scene.get_element_index_from_global_index.return_value = 0

    return scene


class TestFileValidation:
    """Tests for file validation functionality."""

    def test_parser_with_nonexistent_file(self) -> None:
        """Test parser with a nonexistent file.

        Ensures that AnnotationFileNotFoundError is raised for nonexistent files.
        """
        with pytest.raises(AnnotationFileNotFoundError):
            AnnotationParser(Path("nonexistent_file.toml"))

    def test_parser_with_malformed_toml(
        self, malformed_toml_file: str, mock_scene: Any
    ) -> None:
        """Test parser with malformed TOML.

        Args:
            malformed_toml_file: Path to file with malformed TOML content.
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised for files with invalid TOML syntax.
        """
        parser = AnnotationParser(Path(malformed_toml_file), scene=mock_scene)
        with pytest.raises(MalformedAnnotationError):
            parser.parse()

    def test_parser_with_missing_annotations(
        self, missing_annotations_file: str, mock_scene: Any
    ) -> None:
        """Test parser with missing annotations list.

        Args:
            missing_annotations_file: Path to file without annotations list.
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised when the annotations list is missing.
        """
        parser = AnnotationParser(Path(missing_annotations_file), scene=mock_scene)
        with pytest.raises(MalformedAnnotationError):
            parser.parse()

    def test_parser_with_valid_file_no_scene(self, valid_annotations_file: str) -> None:
        """Test parser with a valid annotations file but no scene.

        Args:
            valid_annotations_file: Path to file with valid annotations.

        Ensures that AnnotationError is raised when no scene is provided.
        """
        parser = AnnotationParser(Path(valid_annotations_file))
        with pytest.raises(AnnotationError):
            parser.parse()


class TestAnnotationParsing:
    """Tests for parsing valid annotation files."""

    def test_parser_with_scene(
        self, valid_annotations_file: str, mock_scene: Any
    ) -> None:
        """Test parser with a valid annotations file and scene.

        Args:
            valid_annotations_file: Path to file with valid annotations.
            mock_scene: Mock scene object.

        Ensures that correct annotation objects are created from a valid file.
        """
        parser = AnnotationParser(Path(valid_annotations_file), scene=mock_scene)
        annotations = parser.parse()

        assert len(annotations) == 3

        # Check point annotation
        assert isinstance(annotations[0], PointAnnotation)
        assert annotations[0].label == "Catalytic Site"
        assert len(annotations[0].targets) == 1
        assert annotations[0].indices == [0]  # Mock returns 0 for element index

        # Check pair annotation
        assert isinstance(annotations[1], LineAnnotation)
        assert annotations[1].label == "Binding Pair"
        assert len(annotations[1].targets) == 2
        assert annotations[1].indices == [0, 0]  # Mock returns 0 for element indices

        # Check area annotation
        assert isinstance(annotations[2], AreaAnnotation)
        assert annotations[2].label == "Domain"
        assert len(annotations[2].targets) > 0
        assert annotations[2].indices is None

    def test_discontinuous_range_annotation(self, mock_scene: Any) -> None:
        """Test annotation for a range with discontinuities.

        Args:
            mock_scene: Mock scene object.

        Ensures that area annotations handle ranges with discontinuities correctly.
        """
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


class TestAnnotationValidation:
    """Tests for validation of annotation data."""

    def test_invalid_point_type(self, mock_scene: Any) -> None:
        """Test validation of invalid point type.

        Args:
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised for invalid point annotation data.
        """
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
            parser = AnnotationParser(Path(temp_file), scene=mock_scene)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert "indices" in str(excinfo.value)
        finally:
            os.unlink(temp_file)

    def test_invalid_annotation_type(self, mock_scene: Any) -> None:
        """Test validation of invalid annotation type.

        Args:
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised for unknown annotation types.
        """
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
            parser = AnnotationParser(Path(temp_file), scene=mock_scene)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert "type" in str(excinfo.value)
        finally:
            os.unlink(temp_file)

    def test_invalid_line_indices(self, mock_scene: Any) -> None:
        """Test validation of invalid line indices.

        Args:
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised when line indices are invalid.
        """
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
            parser = AnnotationParser(Path(temp_file), scene=mock_scene)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert "indices" in str(excinfo.value)
            assert "2 indices" in str(excinfo.value)
        finally:
            os.unlink(temp_file)

    def test_invalid_area_range(self, mock_scene: Any) -> None:
        """Test validation of invalid area range.

        Args:
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised when area range is invalid.
        """
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
            parser = AnnotationParser(Path(temp_file), scene=mock_scene)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert "greater than or equal to start" in str(excinfo.value)
        finally:
            os.unlink(temp_file)

    def test_missing_required_field(self, mock_scene: Any) -> None:
        """Test validation of missing required fields.

        Args:
            mock_scene: Mock scene object.

        Ensures that MalformedAnnotationError is raised when required fields are missing.
        """
        content = {
            "annotations": [
                {
                    "label": "Missing Chain",
                    "type": "point",
                    "index": 42,
                    # "chain" is missing
                }
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            toml.dump(content, f)
            temp_file = f.name

        try:
            parser = AnnotationParser(Path(temp_file), scene=mock_scene)
            with pytest.raises(MalformedAnnotationError) as excinfo:
                parser.parse()
            assert "chain" in str(excinfo.value)
        finally:
            os.unlink(temp_file)

    def test_invalid_residue_reference(self, mock_scene: Any) -> None:
        """Test validation of invalid residue reference.

        Args:
            mock_scene: Mock scene object.

        Ensures that InvalidReferenceError is raised for nonexistent residues.
        """
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

    def test_invalid_chain_reference(self, mock_scene: Any) -> None:
        """Test validation of invalid chain reference.

        Args:
            mock_scene: Mock scene object.

        Ensures that InvalidReferenceError is raised for nonexistent chains.
        """
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
