# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scene utility functions."""

from pathlib import Path

import numpy as np

from flatprot.utils.scene import process_structure_chain, process_annotations
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.style import StyleManager
from flatprot.scene import (
    Scene,
    SceneGroup,
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
)
from flatprot.scene.structure import StructureSceneElement


class TestProcessStructureChain:
    """Tests for the process_structure_chain function."""

    def test_process_structure_chain_basic(self, mocker) -> None:
        """Test basic functionality of process_structure_chain.

        Verifies that secondary structure elements are processed correctly and
        added to the scene with proper depth sorting.
        """
        # Mock dependencies
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)
        scene = mocker.Mock(spec=Scene)
        scene.add_element = mocker.Mock()

        # Create mock secondary structure
        ss_element = mocker.Mock()
        ss_element.start = 0
        ss_element.end = 9
        ss_element.secondary_structure_type.value = "helix"

        # Mock chain
        chain = mocker.Mock()
        chain.id = "A"
        chain.secondary_structure = [ss_element]
        chain.num_residues = 10

        # Mock coordinate data
        canvas_coords = np.array([[i, i] for i in range(10)])
        depth_data = np.array([10 - i for i in range(10)])  # Decreasing depth

        coordinate_manager.get.side_effect = lambda start, end, ctype: (
            canvas_coords[start:end]
            if ctype == CoordinateType.CANVAS
            else depth_data[start:end]
            if ctype == CoordinateType.DEPTH
            else None
        )

        # Mock the scene element creation
        mock_element = mocker.Mock(spec=StructureSceneElement)
        mock_element.metadata = {"chain_id": "A", "start": 0, "end": 9, "type": "helix"}

        mocker.patch(
            "flatprot.utils.scene.secondary_structure_to_scene_element",
            return_value=mock_element,
        )

        # Call function
        offset = 0
        new_offset = process_structure_chain(
            chain, offset, coordinate_manager, style_manager, scene
        )

        # Verify results
        assert new_offset == 10, "Should return offset + chain.num_residues"

        # Verify scene.add_element was called exactly twice
        assert scene.add_element.call_count == 2, "Should add group and element"

        # Extract and check the group object from the first call
        chain_group = scene.add_element.call_args_list[0][0][0]
        assert isinstance(chain_group, SceneGroup)
        assert chain_group.id == "A"
        assert chain_group.metadata.get("chain_id") == "A"

        # Check element was added with right arguments
        element_args = scene.add_element.call_args_list[1]
        assert element_args[0][0] == mock_element  # Element object
        assert element_args[0][1] == chain_group  # Parent group
        assert element_args[0][2] == "A"  # Chain ID
        assert element_args[0][3] == 0  # Start index
        assert element_args[0][4] == 9  # End index

    def test_process_structure_chain_multiple_elements(self, mocker) -> None:
        """Test processing multiple secondary structure elements with depth sorting.

        Verifies that elements are correctly sorted by depth before being added to the scene.
        """
        # Mock dependencies
        coordinate_manager = mocker.Mock(spec=CoordinateManager)
        style_manager = mocker.Mock(spec=StyleManager)
        scene = mocker.Mock(spec=Scene)
        scene.add_element = mocker.Mock()

        # Create mock secondary structure elements
        ss_element1 = mocker.Mock()
        ss_element1.start = 0
        ss_element1.end = 9
        ss_element1.secondary_structure_type.value = "helix"

        ss_element2 = mocker.Mock()
        ss_element2.start = 10
        ss_element2.end = 19
        ss_element2.secondary_structure_type.value = "sheet"

        # Mock chain
        chain = mocker.Mock()
        chain.id = "A"
        chain.secondary_structure = [ss_element1, ss_element2]
        chain.num_residues = 20

        # Mock coordinate and depth data
        def mock_get_data(start: int, end: int, ctype: CoordinateType) -> np.ndarray:
            if ctype == CoordinateType.CANVAS:
                return np.array([[i, i] for i in range(start, end)])
            elif ctype == CoordinateType.DEPTH:
                # Make second element have greater depth (further back)
                if start >= 10:
                    return np.ones(end - start) * 20  # Higher value = farther back
                else:
                    return np.ones(end - start) * 10
            return None

        coordinate_manager.get.side_effect = mock_get_data

        # Mock the scene elements
        mock_element1 = mocker.Mock(spec=StructureSceneElement)
        mock_element1.metadata = {
            "chain_id": "A",
            "start": 0,
            "end": 9,
            "type": "helix",
        }

        mock_element2 = mocker.Mock(spec=StructureSceneElement)
        mock_element2.metadata = {
            "chain_id": "A",
            "start": 10,
            "end": 19,
            "type": "sheet",
        }

        mocker.patch(
            "flatprot.utils.scene.secondary_structure_to_scene_element",
            side_effect=[mock_element1, mock_element2],
        )

        # Call function
        offset = 0
        new_offset = process_structure_chain(
            chain, offset, coordinate_manager, style_manager, scene
        )

        # Verify results
        assert new_offset == 20, "Should return offset + chain.num_residues"

        # Verify scene.add_element was called exactly 3 times
        assert scene.add_element.call_count == 3, "Should add group and two elements"

        # Get the chain group from the first call
        chain_group = scene.add_element.call_args_list[0][0][0]
        assert isinstance(chain_group, SceneGroup)
        assert chain_group.id == "A"

        # Extract the elements from the second and third calls
        # Check the depth sorting (higher depth value should be added first)
        second_call_element = scene.add_element.call_args_list[1][0][0]
        third_call_element = scene.add_element.call_args_list[2][0][0]

        # Element2 has higher depth (20) and should be added first
        assert (
            second_call_element == mock_element2
        ), "Element with higher depth should be added first"
        assert (
            third_call_element == mock_element1
        ), "Element with lower depth should be added second"


class TestProcessAnnotations:
    """Tests for the process_annotations function."""

    def test_process_annotations_basic(self, mocker) -> None:
        """Test basic functionality of process_annotations.

        Verifies that annotations are correctly parsed and added to the scene.
        """
        # Mock dependencies
        style_manager = mocker.Mock(spec=StyleManager)
        scene = mocker.Mock(spec=Scene)
        scene.add_element = mocker.Mock()

        # Mock annotation
        mock_annotation = mocker.Mock()
        mock_annotation.label = "Test Annotation"
        mock_annotation.__class__.__name__ = "PointAnnotation"

        # Mock annotation parser
        mock_parser = mocker.Mock()
        mock_parser.parse.return_value = [mock_annotation]

        mocker.patch("flatprot.utils.scene.AnnotationParser", return_value=mock_parser)

        # Mock console
        mocker.patch("flatprot.utils.scene.console.print")

        # Call function
        annotations_path = Path("/path/to/annotations.toml")
        process_annotations(annotations_path, scene, style_manager)

        # Verify AnnotationParser was called correctly
        from flatprot.utils.scene import AnnotationParser

        AnnotationParser.assert_called_once_with(
            file_path=annotations_path,
            scene=scene,
            style_manager=style_manager,
        )

        # Verify annotation was added to scene
        scene.add_element.assert_called_once_with(mock_annotation)

    def test_process_annotations_multiple(self, mocker) -> None:
        """Test processing multiple annotations.

        Verifies that multiple annotations are correctly added to the scene.
        """
        # Mock dependencies
        style_manager = mocker.Mock(spec=StyleManager)
        scene = mocker.Mock(spec=Scene)
        scene.add_element = mocker.Mock()

        # Mock annotations
        mock_annotations = [
            mocker.Mock(label="Annotation 1", spec=PointAnnotation),
            mocker.Mock(label="Annotation 2", spec=LineAnnotation),
            mocker.Mock(label="Annotation 3", spec=AreaAnnotation),
        ]

        # Mock annotation parser
        mock_parser = mocker.Mock()
        mock_parser.parse.return_value = mock_annotations

        mocker.patch("flatprot.utils.scene.AnnotationParser", return_value=mock_parser)

        # Mock console
        mocker.patch("flatprot.utils.scene.console.print")

        # Call function
        annotations_path = Path("/path/to/annotations.toml")
        process_annotations(annotations_path, scene, style_manager)

        # Verify all annotations were added to scene in order
        assert scene.add_element.call_count == 3
        for i, annotation in enumerate(mock_annotations):
            scene.add_element.assert_any_call(annotation)

    def test_process_annotations_error_handling(self, mocker) -> None:
        """Test error handling in process_annotations.

        Verifies that exceptions during annotation processing are caught and reported.
        """
        # Mock dependencies
        style_manager = mocker.Mock(spec=StyleManager)
        scene = mocker.Mock(spec=Scene)
        scene.add_element = mocker.Mock()

        # Mock AnnotationParser to raise an exception
        mocker.patch(
            "flatprot.utils.scene.AnnotationParser",
            side_effect=ValueError("Invalid annotation format"),
        )

        # Mock console
        console_print = mocker.patch("flatprot.utils.scene.console.print")

        # Call function
        annotations_path = Path("/path/to/annotations.toml")
        process_annotations(annotations_path, scene, style_manager)

        # Verify warning was printed
        console_print.assert_called_with(
            "[yellow]Warning: Failed to load annotations: Invalid annotation format[/yellow]"
        )

        # Verify scene.add_element was not called
        assert scene.add_element.call_count == 0
