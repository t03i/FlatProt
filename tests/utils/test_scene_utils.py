# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scene utility functions."""

import pytest
from pytest_mock import MockerFixture
from pathlib import Path
import numpy as np  # Import numpy

# Assuming necessary imports from flatprot.core and flatprot.scene
# You might need to adjust these based on actual object usage
from flatprot.core import (
    Structure,
    Chain,
    ResidueRange,
    ResidueRangeSet,
    SecondaryStructureType,
    CoordinateCalculationError,
)
from flatprot.scene import (
    Scene,
    SceneGroup,
    HelixSceneElement,
    SheetSceneElement,
    BaseAnnotationElement,
    SceneCreationError,
)

# Import the functions to be tested
from flatprot.utils.scene_utils import (
    create_scene_from_structure,
    add_annotations_to_scene,
)

# Import annotation-related classes and errors for mocking/testing
from flatprot.io import (
    AnnotationParser,
    AnnotationError,
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
)

# --- Fixtures ---


# Example fixture for a simple mock structure
@pytest.fixture
def mock_structure(mocker: MockerFixture) -> Structure:
    """Provides a basic mock Structure object."""
    mock_struct = mocker.MagicMock(spec=Structure)
    mock_struct.id = "mock_protein"
    mock_struct.coordinates = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    # Add chains and other necessary attributes as needed for tests
    # mock_struct.__iter__.return_value = [...] # To mock iteration
    return mock_struct


@pytest.fixture
def mock_scene(mocker: MockerFixture) -> Scene:
    """Provides a basic mock Scene object."""
    mock_scn = mocker.MagicMock(spec=Scene)
    mock_scn.id = "mock_scene"
    mock_scn.add_element = mocker.MagicMock()
    return mock_scn


# --- Test Functions ---


def test_create_scene_from_structure_success(mocker: MockerFixture) -> None:
    """Test successful scene creation from a structure with sorting by depth."""
    # 1. Mock Structure, Chains, and SS Elements
    mock_struct = mocker.MagicMock(spec=Structure)
    mock_struct.id = "test_struct"
    mock_struct.coordinates = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    mock_chain_A = mocker.MagicMock(spec=Chain)
    mock_chain_A.id = "A"
    # Use ResidueRange for mock secondary structure elements
    mock_ss_helix = mocker.MagicMock(spec=ResidueRange)
    mock_ss_helix.secondary_structure_type = SecondaryStructureType.HELIX
    mock_ss_helix.start = 10
    mock_ss_helix.end = 20
    mock_ss_sheet = mocker.MagicMock(spec=ResidueRange)
    mock_ss_sheet.secondary_structure_type = SecondaryStructureType.SHEET
    mock_ss_sheet.start = 30
    mock_ss_sheet.end = 35
    # The function iterates over `chain.secondary_structure` which yields these ranges
    mock_chain_A.secondary_structure = [mock_ss_helix, mock_ss_sheet]

    # Make the structure iterable (yields chain_id, chain pairs)
    mock_struct.__iter__.return_value = [("A", mock_chain_A)]

    # 2. Mock Scene Elements and their depth calculation
    mock_scene_class = mocker.patch("flatprot.utils.scene_utils.Scene")
    mock_scene_instance = mocker.MagicMock(spec=Scene)
    mock_scene_instance.add_element = mocker.MagicMock()
    mock_scene_class.return_value = mock_scene_instance

    mock_group_class = mocker.patch("flatprot.utils.scene_utils.SceneGroup")
    mock_group_instance = mocker.MagicMock(spec=SceneGroup)
    mock_group_instance.id = "test_struct_A_group"
    mock_group_class.return_value = mock_group_instance

    # Mock specific element classes *and their get_depth method*
    mock_helix_element = mocker.MagicMock(spec=HelixSceneElement)
    mock_helix_element.id = "helix_A_10-20"
    mock_sheet_element = mocker.MagicMock(spec=SheetSceneElement)
    mock_sheet_element.id = "sheet_A_30-35"

    # Mock get_depth to control sorting order (sheet farther away than helix)
    mock_helix_element.get_depth.return_value = 5.0
    mock_sheet_element.get_depth.return_value = 10.0  # Farther

    # Create mock classes that return the pre-configured elements
    MockHelixClass = mocker.MagicMock(return_value=mock_helix_element)
    MockSheetClass = mocker.MagicMock(return_value=mock_sheet_element)

    # Fix 1: Patch the STRUCTURE_ELEMENT_MAP used inside the function
    # We need the Coil element from the original map for this patch
    from flatprot.utils.scene_utils import STRUCTURE_ELEMENT_MAP

    mock_structure_map = {
        SecondaryStructureType.HELIX: (
            MockHelixClass,
            mocker.MagicMock(),
        ),  # Don't care about style class here
        SecondaryStructureType.SHEET: (MockSheetClass, mocker.MagicMock()),
        SecondaryStructureType.COIL: STRUCTURE_ELEMENT_MAP[
            SecondaryStructureType.COIL
        ],  # Keep original Coil
    }
    mocker.patch.dict(
        "flatprot.utils.scene_utils.STRUCTURE_ELEMENT_MAP", mock_structure_map
    )

    # 3. Call the function under test
    scene = create_scene_from_structure(mock_struct)

    # 4. Assertions
    assert scene == mock_scene_instance
    mock_scene_class.assert_called_once_with(structure=mock_struct)

    # Check group creation and addition
    mock_group_class.assert_called_once_with(id="test_struct_A")
    mock_scene_instance.add_element.assert_any_call(mock_group_instance)

    # Check element creation calls (using the mocked classes from the map)
    expected_helix_range = ResidueRangeSet(
        [ResidueRange(chain_id="A", start=10, end=20)]
    )
    MockHelixClass.assert_called_once_with(
        residue_range_set=expected_helix_range, style=None
    )
    expected_sheet_range = ResidueRangeSet(
        [ResidueRange(chain_id="A", start=30, end=35)]
    )
    MockSheetClass.assert_called_once_with(
        residue_range_set=expected_sheet_range, style=None
    )

    # Check get_depth calls on the *instances* returned by the mock classes
    mock_helix_element.get_depth.assert_called_once_with(mock_struct)
    mock_sheet_element.get_depth.assert_called_once_with(mock_struct)

    # Check elements were added to the scene *with the group as parent*
    add_calls = mock_scene_instance.add_element.call_args_list
    assert len(add_calls) == 4  # 3 elements + 1 connection
    assert add_calls[0] == mocker.call(mock_group_instance)
    assert add_calls[2] == mocker.call(
        mock_sheet_element, parent_id=mock_group_instance.id
    )  # Sheet (10.0) added first
    assert add_calls[3] == mocker.call(
        mock_helix_element, parent_id=mock_group_instance.id
    )  # Helix (5.0) added second


def test_create_scene_from_structure_no_coords(mock_structure: Structure) -> None:
    """Test scene creation raises error if structure has no coordinates."""
    mock_structure.coordinates = None
    with pytest.raises(SceneCreationError, match="has no coordinates"):
        create_scene_from_structure(mock_structure)

    mock_structure.coordinates = []  # Also test with empty list/array
    with pytest.raises(SceneCreationError, match="has no coordinates"):
        create_scene_from_structure(mock_structure)


def test_create_scene_from_structure_depth_error(mocker: MockerFixture) -> None:
    """Test scene creation handles CoordinateCalculationError during depth calculation."""
    # 1. Set up mock structure
    mock_struct = mocker.MagicMock(spec=Structure)
    mock_struct.id = "test_struct_depth_err"
    mock_struct.coordinates = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mock_chain_A = mocker.MagicMock(spec=Chain)
    mock_chain_A.id = "A"
    mock_ss_helix = mocker.MagicMock(spec=ResidueRange)
    mock_ss_helix.secondary_structure_type = SecondaryStructureType.HELIX
    mock_ss_helix.start = 10
    mock_ss_helix.end = 20
    mock_chain_A.secondary_structure = [mock_ss_helix]
    mock_struct.__iter__.return_value = [("A", mock_chain_A)]

    mocker.patch("flatprot.utils.scene_utils.Scene")
    mocker.patch("flatprot.utils.scene_utils.SceneGroup")

    # Mock Helix Element instance and its get_depth method
    mock_helix_element = mocker.MagicMock(spec=HelixSceneElement)
    mock_helix_element.id = "helix_A_10-20_err"
    depth_error_msg = "Depth calculation failed!"
    mock_helix_element.get_depth.side_effect = CoordinateCalculationError(
        depth_error_msg
    )

    # Mock Helix Class to return the faulty element instance
    MockHelixClass = mocker.MagicMock(return_value=mock_helix_element)

    # Fix 1: Patch the STRUCTURE_ELEMENT_MAP
    from flatprot.utils.scene_utils import STRUCTURE_ELEMENT_MAP

    mock_structure_map = {
        SecondaryStructureType.HELIX: (MockHelixClass, mocker.MagicMock()),
        SecondaryStructureType.SHEET: STRUCTURE_ELEMENT_MAP[
            SecondaryStructureType.SHEET
        ],
        SecondaryStructureType.COIL: STRUCTURE_ELEMENT_MAP[SecondaryStructureType.COIL],
    }
    mocker.patch.dict(
        "flatprot.utils.scene_utils.STRUCTURE_ELEMENT_MAP", mock_structure_map
    )

    # 3. Call create_scene_from_structure
    with pytest.raises(SceneCreationError) as excinfo:
        create_scene_from_structure(mock_struct)

    # 4. Assert SceneCreationError is raised wrapping the original error
    # Fix 2: Assert message is in the *cause*
    assert depth_error_msg in str(excinfo.value.__cause__)
    assert isinstance(excinfo.value.__cause__, CoordinateCalculationError)

    # Verify element constructor was called
    expected_helix_range = ResidueRangeSet(
        [ResidueRange(chain_id="A", start=10, end=20)]
    )
    MockHelixClass.assert_called_once_with(
        residue_range_set=expected_helix_range, style=None
    )
    # Verify get_depth was called
    mock_helix_element.get_depth.assert_called_once_with(mock_struct)


def test_create_scene_from_structure_element_creation_error(
    mocker: MockerFixture,
) -> None:
    """Test scene creation handles unexpected errors during element instantiation."""
    # 1. Set up mock structure
    mock_struct = mocker.MagicMock(spec=Structure)
    mock_struct.id = "test_struct_create_err"
    mock_struct.coordinates = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mock_chain_A = mocker.MagicMock(spec=Chain)
    mock_chain_A.id = "A"
    mock_ss_helix = mocker.MagicMock(spec=ResidueRange)
    mock_ss_helix.secondary_structure_type = SecondaryStructureType.HELIX
    mock_ss_helix.start = 10
    mock_ss_helix.end = 20
    mock_chain_A.secondary_structure = [mock_ss_helix]
    mock_struct.__iter__.return_value = [("A", mock_chain_A)]

    mocker.patch("flatprot.utils.scene_utils.Scene")
    mocker.patch("flatprot.utils.scene_utils.SceneGroup")

    # 2. Mock the Helix element constructor (class) to raise an unexpected error
    creation_error_msg = "Something went wrong during creation!"
    MockHelixClass = mocker.MagicMock(side_effect=ValueError(creation_error_msg))

    # Fix 1 & 3: Patch the STRUCTURE_ELEMENT_MAP
    from flatprot.utils.scene_utils import STRUCTURE_ELEMENT_MAP

    mock_structure_map = {
        SecondaryStructureType.HELIX: (MockHelixClass, mocker.MagicMock()),
        SecondaryStructureType.SHEET: STRUCTURE_ELEMENT_MAP[
            SecondaryStructureType.SHEET
        ],
        SecondaryStructureType.COIL: STRUCTURE_ELEMENT_MAP[SecondaryStructureType.COIL],
    }
    mocker.patch.dict(
        "flatprot.utils.scene_utils.STRUCTURE_ELEMENT_MAP", mock_structure_map
    )

    # 3. Call create_scene_from_structure
    with pytest.raises(SceneCreationError) as excinfo:
        create_scene_from_structure(mock_struct)

    # 4. Assert SceneCreationError is raised wrapping the original error
    assert "Unexpected error creating element" in str(excinfo.value)
    assert creation_error_msg in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)

    # Verify the constructor was called (which raised the error)
    expected_helix_range = ResidueRangeSet(
        [ResidueRange(chain_id="A", start=10, end=20)]
    )
    MockHelixClass.assert_called_once_with(
        residue_range_set=expected_helix_range, style=None
    )


def test_add_annotations_to_scene_success(
    mocker: MockerFixture, mock_scene: Scene
) -> None:
    """Test successfully adding annotations from a mocked parser."""
    # Mock Path.exists needed if __init__ did checks, but AnnotationParser doesn't anymore
    # mocker.patch("pathlib.Path.exists", return_value=True) # Keep if AnnotationParser.__init__ checks again

    # Prepare mock annotations
    mock_anno1 = mocker.MagicMock(
        spec=BaseAnnotationElement,
        id="anno1",
        # Provide a class with __name__ for potential logging/debugging within the tested function
        __class__=type(
            "MockAnno1", (BaseAnnotationElement,), {"__name__": "MockAnno1"}
        ),
    )
    mock_anno2 = mocker.MagicMock(
        spec=BaseAnnotationElement,
        id="anno2",
        __class__=type(
            "MockAnno2", (BaseAnnotationElement,), {"__name__": "MockAnno2"}
        ),
    )

    # --- Mocking Strategy Change ---
    # 1. Mock the AnnotationParser class *where it's imported* in scene_utils
    MockAnnotationParserClass = mocker.patch(
        "flatprot.utils.scene_utils.AnnotationParser"
    )

    # 2. Configure the mock *instance* that the mocked class will return
    mock_parser_instance = mocker.MagicMock(spec=AnnotationParser)
    # Configure the parse method on the mock instance
    mock_parser_instance.parse.return_value = [mock_anno1, mock_anno2]
    # Make the mocked class return our configured instance when called
    MockAnnotationParserClass.return_value = mock_parser_instance
    # --- End Mocking Strategy Change ---

    dummy_path = Path("dummy_annotations.toml")

    # Call the function under test
    add_annotations_to_scene(dummy_path, mock_scene)

    # Assert AnnotationParser class was instantiated correctly
    MockAnnotationParserClass.assert_called_once_with(dummy_path)

    # Assert the parse method was called on the *instance*
    mock_parser_instance.parse.assert_called_once_with()

    # Assert mock_scene.add_element was called for each annotation object
    assert mock_scene.add_element.call_count == 2
    mock_scene.add_element.assert_has_calls(
        [mocker.call(mock_anno1), mocker.call(mock_anno2)], any_order=False
    )


def test_add_annotations_to_scene_parser_errors(
    mocker: MockerFixture, mock_scene: Scene
) -> None:
    """Test that errors from AnnotationParser are correctly raised."""
    # Mock Path.exists needed if __init__ did checks, but AnnotationParser doesn't anymore
    # mocker.patch("pathlib.Path.exists", return_value=True) # Keep if AnnotationParser.__init__ checks again

    # --- Mocking Strategy Change ---
    # Mock the AnnotationParser class where it's imported
    MockAnnotationParserClass = mocker.patch(
        "flatprot.utils.scene_utils.AnnotationParser"
    )
    mock_parser_instance = mocker.MagicMock(spec=AnnotationParser)
    MockAnnotationParserClass.return_value = mock_parser_instance
    # --- End Mocking Strategy Change ---

    dummy_path = Path("dummy_annotations.toml")

    # Test FileNotFoundError raised from *parse* side_effect
    mock_parser_instance.parse.side_effect = AnnotationFileNotFoundError(
        "File not found"
    )
    with pytest.raises(AnnotationFileNotFoundError):
        add_annotations_to_scene(dummy_path, mock_scene)
    MockAnnotationParserClass.assert_called_with(dummy_path)  # Check instantiation
    mock_parser_instance.parse.assert_called_once()  # Check parse call
    mock_parser_instance.parse.reset_mock()  # Reset for next test
    mock_parser_instance.parse.side_effect = None
    MockAnnotationParserClass.reset_mock()  # Reset class mock too

    # Test MalformedAnnotationError
    mock_parser_instance.parse.side_effect = MalformedAnnotationError(
        "test_file.toml", "Bad format"
    )
    with pytest.raises(MalformedAnnotationError):
        add_annotations_to_scene(dummy_path, mock_scene)
    MockAnnotationParserClass.assert_called_with(dummy_path)
    mock_parser_instance.parse.assert_called_once()
    mock_parser_instance.parse.reset_mock()
    mock_parser_instance.parse.side_effect = None
    MockAnnotationParserClass.reset_mock()

    # Test generic AnnotationError
    mock_parser_instance.parse.side_effect = AnnotationError("Generic parse error")
    with pytest.raises(AnnotationError):
        add_annotations_to_scene(dummy_path, mock_scene)
    MockAnnotationParserClass.assert_called_with(dummy_path)
    mock_parser_instance.parse.assert_called_once()
    mock_parser_instance.parse.reset_mock()
    mock_parser_instance.parse.side_effect = None
    MockAnnotationParserClass.reset_mock()

    # Test unexpected error during parsing
    mock_parser_instance.parse.side_effect = ValueError("Unexpected parse issue")
    with pytest.raises(ValueError):  # Should re-raise the original unexpected error
        add_annotations_to_scene(dummy_path, mock_scene)
    MockAnnotationParserClass.assert_called_with(dummy_path)
    mock_parser_instance.parse.assert_called_once()


def test_add_annotations_to_scene_add_element_error(
    mocker: MockerFixture, mock_scene: Scene
) -> None:
    """Test that errors during scene.add_element are handled."""
    # Mock Path.exists needed if __init__ did checks, but AnnotationParser doesn't anymore
    # mocker.patch("pathlib.Path.exists", return_value=True) # Keep if AnnotationParser.__init__ checks again

    # --- Mocking Strategy Change ---
    # Mock AnnotationParser class
    MockAnnotationParserClass = mocker.patch(
        "flatprot.utils.scene_utils.AnnotationParser"
    )
    mock_parser_instance = mocker.MagicMock(spec=AnnotationParser)
    MockAnnotationParserClass.return_value = mock_parser_instance

    # Configure mock instance's parse method
    mock_annotation = mocker.MagicMock(spec=BaseAnnotationElement)
    mock_annotation.id = "anno1"
    # Set __class__ to provide __name__ for the error message format
    mock_annotation.__class__ = type(
        "MockAnno", (BaseAnnotationElement,), {"__name__": "MockAnno"}
    )
    mock_parser_instance.parse.return_value = [mock_annotation]
    # --- End Mocking Strategy Change ---

    # Mock scene.add_element to raise an error
    mock_scene.add_element.side_effect = Exception("Failed to add element")

    dummy_path = Path("dummy_annotations.toml")

    with pytest.raises(SceneCreationError, match="Failed to add annotation 'anno1'"):
        add_annotations_to_scene(dummy_path, mock_scene)

    # Assert AnnotationParser was instantiated
    MockAnnotationParserClass.assert_called_once_with(dummy_path)
    # Assert parse was called on the instance
    mock_parser_instance.parse.assert_called_once_with()
    # Assert add_element was called
    mock_scene.add_element.assert_called_once_with(mock_annotation)
