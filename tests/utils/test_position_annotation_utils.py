# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for position annotation utility functions."""

import pytest

from flatprot.core import (
    Structure,
    ResidueRangeSet,
)
from flatprot.scene import (
    Scene,
    HelixSceneElement,
    SheetSceneElement,
    CoilSceneElement,
    PositionAnnotationStyle,
    PositionType,
    SceneCreationError,
)
from flatprot.utils.scene_utils import add_position_annotations_to_scene


class TestAddPositionAnnotationsToScene:
    """Test the add_position_annotations_to_scene utility function."""

    def create_mock_scene(self, mocker):
        """Create a mock scene with structure elements."""
        mock_structure = mocker.MagicMock(spec=Structure)
        scene = Scene(structure=mock_structure)

        # Create mock structure elements
        helix1 = mocker.MagicMock(spec=HelixSceneElement)
        helix1.id = "helix_1"
        helix1.residue_range_set = ResidueRangeSet.from_string("A:10-20")

        sheet1 = mocker.MagicMock(spec=SheetSceneElement)
        sheet1.id = "sheet_1"
        sheet1.residue_range_set = ResidueRangeSet.from_string("A:30-40")

        coil1 = mocker.MagicMock(spec=CoilSceneElement)
        coil1.id = "coil_1"
        coil1.residue_range_set = ResidueRangeSet.from_string("A:21-29")

        # Mock get_sequential_structure_elements to return elements in order
        mock_elements = [helix1, coil1, sheet1]  # Ordered by sequence
        scene.get_sequential_structure_elements = mocker.MagicMock(
            return_value=mock_elements
        )

        # Mock add_element method
        scene.add_element = mocker.MagicMock()

        return scene, mock_elements

    def test_add_position_annotations_default(self, mocker):
        """Test adding position annotations with default settings."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Call the function with default annotation level (full)
        add_position_annotations_to_scene(scene, annotation_level="full")

        # Verify add_element was called
        assert scene.add_element.call_count > 0

        # Check that N and C terminus annotations were added
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]

        # Should have N-terminus, C-terminus, and residue numbers for helix and sheet
        n_terminus = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.N_TERMINUS
        ]
        c_terminus = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.C_TERMINUS
        ]
        residue_numbers = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.RESIDUE_NUMBER
        ]

        assert len(n_terminus) == 1
        assert len(c_terminus) == 1
        assert (
            len(residue_numbers) == 4
        )  # 2 for helix (start, end) + 2 for sheet (start, end)

    def test_add_position_annotations_terminus_only(self, mocker):
        """Test adding only terminus annotations."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Call with minimal level (terminus only)
        add_position_annotations_to_scene(scene, annotation_level="minimal")

        # Check added annotations
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]

        n_terminus = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.N_TERMINUS
        ]
        c_terminus = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.C_TERMINUS
        ]
        residue_numbers = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.RESIDUE_NUMBER
        ]

        assert len(n_terminus) == 1
        assert len(c_terminus) == 1
        assert len(residue_numbers) == 0  # Should be none

    def test_add_position_annotations_residue_numbers_only(self, mocker):
        """Test adding residue number annotations with terminus."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Call with major level (terminus + major structures)
        add_position_annotations_to_scene(scene, annotation_level="major")

        # Check added annotations
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]

        n_terminus = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.N_TERMINUS
        ]
        c_terminus = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.C_TERMINUS
        ]
        residue_numbers = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.RESIDUE_NUMBER
        ]

        assert len(n_terminus) == 1  # Should have terminus
        assert len(c_terminus) == 1  # Should have terminus
        assert len(residue_numbers) == 4  # 2 for helix + 2 for sheet

    def test_add_position_annotations_no_annotations(self, mocker):
        """Test that no annotations are added when level is none."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Call with none level
        add_position_annotations_to_scene(scene, annotation_level="none")

        # Check added annotations
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]

        assert len(added_annotations) == 0

    def test_add_position_annotations_custom_style(self, mocker):
        """Test adding position annotations with custom style."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Create custom style
        custom_style = PositionAnnotationStyle(
            font_size=12.0,
            terminus_font_size=16.0,
        )

        # Call with custom style
        add_position_annotations_to_scene(
            scene, style=custom_style, annotation_level="full"
        )

        # Check that annotations were added with custom style
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]

        # All annotations should have the custom style
        for annotation in added_annotations:
            assert annotation.style.font_size == 12.0
            assert annotation.style.terminus_font_size == 16.0

    def test_add_position_annotations_empty_scene(self, mocker):
        """Test handling of scene with no structure elements."""
        mock_structure = mocker.MagicMock(spec=Structure)
        scene = Scene(structure=mock_structure)
        scene.get_sequential_structure_elements = mocker.MagicMock(return_value=[])
        scene.add_element = mocker.MagicMock()

        # Should not raise error but also not add any annotations
        add_position_annotations_to_scene(scene, annotation_level="full")

        # Should not have called add_element
        assert scene.add_element.call_count == 0

    def test_add_position_annotations_get_elements_error(self, mocker):
        """Test handling of error when getting structure elements."""
        mock_structure = mocker.MagicMock(spec=Structure)
        scene = Scene(structure=mock_structure)
        scene.get_sequential_structure_elements = mocker.MagicMock(
            side_effect=Exception("Test error")
        )

        # Should raise SceneCreationError
        with pytest.raises(
            SceneCreationError, match="Failed to get structure elements"
        ):
            add_position_annotations_to_scene(scene, annotation_level="full")

    def test_add_position_annotations_add_element_error(self, mocker):
        """Test handling of error when adding annotation to scene."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Make add_element raise an error for N-terminus
        def add_element_side_effect(element):
            if (
                hasattr(element, "position_type")
                and element.position_type == PositionType.N_TERMINUS
            ):
                raise Exception("Test add error")

        scene.add_element.side_effect = add_element_side_effect

        # Should raise SceneCreationError for N-terminus
        with pytest.raises(
            SceneCreationError, match="Failed to add N-terminus annotation"
        ):
            add_position_annotations_to_scene(scene, annotation_level="full")

    def test_residue_number_annotations_skip_coils(self, mocker):
        """Test that residue number annotations are not added for coil elements."""
        scene, mock_elements = self.create_mock_scene(mocker)

        # Call function with major level (includes residue numbers)
        add_position_annotations_to_scene(scene, annotation_level="major")

        # Check that only helix and sheet get residue numbers, not coil
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]
        residue_number_annotations = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.RESIDUE_NUMBER
        ]

        # Should be 4 annotations: 2 for helix + 2 for sheet (no coil)
        assert len(residue_number_annotations) == 4

    def test_residue_number_single_residue_element(self, mocker):
        """Test handling of single-residue elements (no end annotation)."""
        mock_structure = mocker.MagicMock(spec=Structure)
        scene = Scene(structure=mock_structure)

        # Create a single-residue helix
        helix1 = mocker.MagicMock(spec=HelixSceneElement)
        helix1.id = "single_helix"
        helix1.residue_range_set = ResidueRangeSet.from_string(
            "A:42-42"
        )  # Single residue

        mock_elements = [helix1]
        scene.get_sequential_structure_elements = mocker.MagicMock(
            return_value=mock_elements
        )
        scene.add_element = mocker.MagicMock()

        # Call function
        add_position_annotations_to_scene(scene, annotation_level="full")

        # Should only add start annotation, not end (since start == end)
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]
        residue_number_annotations = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.RESIDUE_NUMBER
        ]

        # Should only be 1 annotation for the single residue
        assert len(residue_number_annotations) == 1

    def test_element_with_no_ranges(self, mocker):
        """Test handling of structure element with no residue ranges."""
        mock_structure = mocker.MagicMock(spec=Structure)
        scene = Scene(structure=mock_structure)

        # Create element with empty range set
        helix1 = mocker.MagicMock(spec=HelixSceneElement)
        helix1.id = "empty_helix"
        helix1.residue_range_set = ResidueRangeSet([])  # Empty ranges

        mock_elements = [helix1]
        scene.get_sequential_structure_elements = mocker.MagicMock(
            return_value=mock_elements
        )
        scene.add_element = mocker.MagicMock()

        # Call function - should not raise error
        add_position_annotations_to_scene(scene, annotation_level="full")

        # Should not have added any residue number annotations
        calls = scene.add_element.call_args_list
        added_annotations = [call[0][0] for call in calls]
        residue_number_annotations = [
            ann
            for ann in added_annotations
            if ann.position_type == PositionType.RESIDUE_NUMBER
        ]

        assert len(residue_number_annotations) == 0
