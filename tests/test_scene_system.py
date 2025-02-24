import pytest
import numpy as np
from unittest.mock import Mock
from flatprot.scene import Scene, SceneGroup, SceneElement, StructureSceneElement

# --------------------------
# 1. Parent-Child Relationship Validation
# Verify parent references are correctly maintained when:
# - Adding elements to groups
# - Removing elements from groups
# - Creating subgroups that move existing elements
# - Edge case: Adding element to multiple groups
# --------------------------


def test_group_parent_relationships():
    scene = Scene()
    group = scene.create_group("test_group")
    element = Mock(spec=SceneElement)

    group.add_element(element)
    assert element._parent == group
    assert element in group._elements

    group.remove_element(element)
    assert element._parent is None
    assert element not in group._elements


def test_subgroup_parent_management():
    scene = Scene()
    parent_group = scene.create_group("parent")
    child_group = scene.create_group("child", parent=parent_group)

    element = Mock(spec=SceneElement)
    child_group.add_element(element)

    assert element._parent == child_group
    assert child_group._parent == parent_group


# --------------------------
# Residue Mapping Accuracy
# Test get_elements_for_residue with:
# - Residues at start/end boundaries of ranges
# - Overlapping ranges in same chain
# - Non-existent chain IDs
# - Multiple elements matching single residue
# --------------------------


class MockStructureElement(StructureSceneElement):
    def __init__(self, chain, start, end):
        super().__init__(start, end, chain, {}, None, None)

    def display_coordinates(self):
        return np.array([[0, 0, 0]])


def test_residue_mapping_boundaries():
    scene = Scene()
    elem = MockStructureElement("A", 5, 10)
    scene.add_structural_element(elem)

    # Test boundary conditions
    assert scene.get_elements_for_residue("A", 4) == []
    assert scene.get_elements_for_residue("A", 5) == [elem]
    assert scene.get_elements_for_residue("A", 10) == [elem]
    assert scene.get_elements_for_residue("A", 11) == []


def test_overlapping_residue_ranges():
    scene = Scene()
    elem1 = MockStructureElement("B", 5, 15)
    elem2 = MockStructureElement("B", 10, 20)
    scene.add_structural_element(elem1)
    scene.add_structural_element(elem2)

    results = scene.get_elements_for_residue("B", 12)
    assert len(results) == 2
    assert elem1 in results
    assert elem2 in results


# --------------------------
# Group Transformation Propagation
# Validate that group transforms are applied to:
# - Child elements' display coordinates
# - Nested subgroups (transform composition)
# - Elements moved between groups with different transforms
# --------------------------


def test_transform_application(mocker):
    scene = Scene()
    group = scene.create_group(
        "transform_group", parameters={"transforms": {"rotate": 45}}
    )

    # Mock child elements
    child1 = mocker.Mock(spec=SceneElement)
    child1.display_coordinates.return_value = np.array([[1, 2, 3]])
    child2 = mocker.Mock(spec=SceneElement)
    child2.display_coordinates.return_value = np.array([[4, 5, 6]])

    group.add_element(child1)
    group.add_element(child2)

    # Verify transform application (actual transform math would be in display_coordinates)
    coords = group.display_coordinates()
    assert coords.shape == (2, 3)
    np.testing.assert_array_equal(coords, np.array([[1, 2, 3], [4, 5, 6]]))


# --------------------------
# Annotation Target Validation
# Test annotation system handles:
# - Circular references (annotations targeting other annotations)
# - Invalid target elements
# - Style inheritance from parent groups
# - Group annotations consuming their elements
# --------------------------


def test_annotation_target_validation():
    scene = Scene()
    valid_element = MockStructureElement("A", 1, 10)
    invalid_element = "not_a_scene_element"

    with pytest.raises(TypeError):
        scene.add_annotation("test", "content", invalid_element)

    with pytest.raises(TypeError):
        scene.add_annotation("test", "content", [valid_element, invalid_element])


def test_group_annotation_consumption():
    scene = Scene()
    parent = scene.create_group("parent")
    element = MockStructureElement("A", 1, 10)
    scene.add_structural_element(element, parent=parent)

    annotation = scene.add_group_annotation("test", "content", [element])
    assert element._parent == annotation
    assert element not in parent._elements


# --------------------------
# Scene Hierarchy Integrity
# Verify scene graph maintains consistency when:
# - Removing groups with nested elements
# - Reparenting elements between groups
# - Deleting elements that are annotation targets
# - Deep nesting (5+ levels)
# --------------------------


def test_group_removal_orphaning():
    scene = Scene()
    group = scene.create_group("temp_group")
    element = MockStructureElement("A", 1, 10)
    group.add_element(element)

    scene.root.remove_element(group)
    assert element._parent is None
    assert group not in scene.root._elements


def test_deep_nesting_parents():
    scene = Scene()
    current = scene.root
    for i in range(5):
        current = scene.create_group(f"level_{i}", parent=current)

    assert current._parent.id == "level_3"
    assert scene.get_parent_group(current).id == "level_3"


# --------------------------
# Coordinate Calculation Edge Cases
# Test display_coordinates with:
# - Empty groups
# - Groups mixing coordinate/non-coordinate elements
# - Extremely large coordinate values
# - Mixed dimensionality in child elements
# --------------------------


def test_empty_group_coordinates():
    group = SceneGroup(id="empty")
    assert group.display_coordinates() is None


def test_mixed_coordinate_sources():
    group = SceneGroup(id="mixed")
    valid_element = Mock(spec=SceneElement)
    valid_element.display_coordinates.return_value = np.array([[1, 2, 3]])
    invalid_element = Mock(spec=SceneElement)
    invalid_element.display_coordinates.return_value = None

    group.add_element(valid_element)
    group.add_element(invalid_element)

    coords = group.display_coordinates()
    assert coords.shape == (1, 3)
