import pytest
import numpy as np
from unittest.mock import Mock

from flatprot.scene import Scene, SceneGroup, SceneElement, StructureSceneElement


# --------------------------
# 1. Element Registration Tests
# --------------------------


def test_element_registration(mocker):
    """Test basic element registration with and without residue mapping."""
    scene = Scene()
    element = mocker.Mock(spec=SceneElement)

    # Basic registration
    scene.add_element(element)
    assert element in scene._elements
    assert element._parent == scene.root

    # Registration with residue mapping
    struct_element = mocker.Mock(spec=StructureSceneElement)
    scene.add_element(struct_element, chain_id="A", start=1, end=10)
    assert struct_element in scene._elements
    assert struct_element in scene._residue_mappings


def test_element_unregistration(mocker):
    """Test element unregistration preserves residue mappings."""
    scene = Scene()
    element = mocker.Mock(spec=StructureSceneElement)
    scene.add_element(element, chain_id="A", start=1, end=10)

    mapping = scene._unregister_element(element)
    assert element not in scene._elements
    assert mapping is not None
    assert mapping.chain_id == "A"
    assert mapping.start == 1
    assert mapping.end == 10


# --------------------------
# 2. Parent-Child Relationship Tests
# --------------------------


def test_parent_child_management(mocker):
    """Test parent-child relationship management."""
    scene = Scene()
    # Create mocks with _elements attribute
    parent = mocker.Mock(spec=SceneGroup)
    parent._elements = []
    parent.add_element = mocker.Mock()
    parent.remove_element = mocker.Mock()

    child = mocker.Mock(spec=SceneGroup)
    child._elements = []
    child.add_element = mocker.Mock()
    child.remove_element = mocker.Mock()

    # Test setting parent
    scene.add_element(parent)
    scene.add_element(child, parent=parent)

    assert child._parent == parent
    parent.add_element.assert_called_with(child)


def test_move_element_to_parent(mocker):
    """Test moving elements between parents."""
    scene = Scene()
    # Create mocks with _elements attribute
    old_parent = mocker.Mock(spec=SceneGroup)
    old_parent._elements = []
    old_parent.add_element = mocker.Mock()
    old_parent.remove_element = mocker.Mock()

    new_parent = mocker.Mock(spec=SceneGroup)
    new_parent._elements = []
    new_parent.add_element = mocker.Mock(
        side_effect=lambda element: setattr(element, "_parent", new_parent)
    )
    new_parent.remove_element = mocker.Mock()

    element = mocker.Mock(spec=SceneElement)
    element._parent = None

    # Test initial parent assignment
    scene.add_element(old_parent)
    scene.add_element(element, parent=old_parent)
    assert element._parent == old_parent, "Element's parent should be set to old parent"

    # Test moving to new parent
    scene.add_element(new_parent)
    scene.move_element_to_parent(element, new_parent)
    assert element._parent == new_parent, "Element's parent should be set to new parent"

    old_parent.remove_element.assert_called_once()
    new_parent.add_element.assert_called_once()
    new_parent.add_element.assert_called_with(element)
    old_parent.remove_element.assert_called_with(element)


# --------------------------
# 3. Residue Mapping Tests
# --------------------------


def test_residue_mapping_queries():
    """Test querying elements by residue position."""
    scene = Scene()
    element1 = Mock(spec=StructureSceneElement)
    element2 = Mock(spec=StructureSceneElement)

    scene.add_element(element1, chain_id="A", start=1, end=10)
    scene.add_element(element2, chain_id="A", start=5, end=15)

    assert len(list(scene)) == 2, "Scene should have 2 elements"

    # Test overlapping region
    elements = scene.get_elements_for_residue("A", 7)
    assert len(elements) == 2, "Should identify 2 elements for the residue query"
    assert element1 in elements
    assert element2 in elements

    # Test boundary conditions
    assert len(scene.get_elements_for_residue("A", 1)) == 1
    assert len(scene.get_elements_for_residue("A", 15)) == 1
    assert len(scene.get_elements_for_residue("A", 20)) == 0


def test_invalid_residue_queries():
    """Test handling of invalid residue queries."""
    scene = Scene()

    # Test non-existent chain
    assert scene.get_elements_for_residue("X", 1) == []

    # Test invalid residue number
    element = Mock(spec=StructureSceneElement)
    scene.add_element(element, chain_id="A", start=1, end=10)
    assert scene.get_elements_for_residue("A", 0) == []
    assert scene.get_elements_for_residue("A", 11) == []


def test_residue_range_queries():
    """Test querying elements by residue range with different overlap scenarios."""
    scene = Scene()
    element1 = Mock(spec=StructureSceneElement)
    element2 = Mock(spec=StructureSceneElement)
    element3 = Mock(spec=StructureSceneElement)

    # Set up elements with different ranges
    scene.add_element(element1, chain_id="A", start=1, end=10)  # [1-10]
    scene.add_element(element2, chain_id="A", start=5, end=15)  # [5-15]
    scene.add_element(element3, chain_id="A", start=20, end=30)  # [20-30]

    # Test complete overlap
    elements = scene.get_elements_for_residue_range("A", 7, 8)
    assert len(elements) == 2
    assert element1 in elements
    assert element2 in elements

    # Test partial overlap at start
    elements = scene.get_elements_for_residue_range("A", 3, 7)
    assert len(elements) == 2
    assert element1 in elements
    assert element2 in elements

    # Test partial overlap at end
    elements = scene.get_elements_for_residue_range("A", 8, 12)
    assert len(elements) == 2
    assert element1 in elements
    assert element2 in elements

    # Test range containing an element
    elements = scene.get_elements_for_residue_range("A", 4, 16)
    assert len(elements) == 2
    assert element1 in elements
    assert element2 in elements

    # Test non-overlapping range
    elements = scene.get_elements_for_residue_range("A", 16, 19)
    assert len(elements) == 0

    # Test range between elements
    elements = scene.get_elements_for_residue_range("A", 16, 19)
    assert len(elements) == 0


def test_residue_range_edge_cases():
    """Test edge cases for residue range queries."""
    scene = Scene()
    element = Mock(spec=StructureSceneElement)
    scene.add_element(element, chain_id="A", start=10, end=20)

    # Test exact bounds
    elements = scene.get_elements_for_residue_range("A", 10, 20)
    assert len(elements) == 1
    assert element in elements

    # Test invalid chain
    elements = scene.get_elements_for_residue_range("B", 10, 20)
    assert len(elements) == 0

    # Test invalid range (start > end)
    elements = scene.get_elements_for_residue_range("A", 20, 10)
    assert len(elements) == 0

    # Test single residue range
    elements = scene.get_elements_for_residue_range("A", 15, 15)
    assert len(elements) == 1
    assert element in elements


def test_multiple_chain_residue_ranges():
    """Test residue range queries across multiple chains."""
    scene = Scene()
    element_a = Mock(spec=StructureSceneElement)
    element_b = Mock(spec=StructureSceneElement)

    scene.add_element(element_a, chain_id="A", start=1, end=10)
    scene.add_element(element_b, chain_id="B", start=1, end=10)

    # Test same range in different chains
    elements_a = scene.get_elements_for_residue_range("A", 1, 5)
    elements_b = scene.get_elements_for_residue_range("B", 1, 5)

    assert len(elements_a) == 1
    assert len(elements_b) == 1
    assert element_a in elements_a
    assert element_b in elements_b
    assert element_a not in elements_b
    assert element_b not in elements_a


def test_global_to_local_index_mapping():
    """Test conversion from global residue indices to element-local indices."""
    scene = Scene()
    element = Mock(spec=StructureSceneElement)

    # Register element with residues 10-20
    scene.add_element(element, chain_id="A", start=10, end=20)

    # Test various index conversions
    assert scene.get_element_index_from_global_index(10, element) == 0  # First residue
    assert scene.get_element_index_from_global_index(15, element) == 5  # Middle residue
    assert scene.get_element_index_from_global_index(20, element) == 10  # Last residue

    # Test invalid element
    invalid_element = Mock(spec=StructureSceneElement)
    with pytest.raises(AssertionError):
        scene.get_element_index_from_global_index(10, invalid_element)

    # Test global index outside element range
    with pytest.raises(AssertionError):
        scene.get_element_index_from_global_index(21, element)


# --------------------------
# 4. Group Management Tests
# --------------------------


def test_move_elements_to_group():
    """Test moving multiple elements to a new group."""
    scene = Scene()
    elements = [Mock(spec=SceneElement) for _ in range(3)]
    new_group = SceneGroup(id="new_group")

    # Add elements to scene root
    for element in elements:
        scene.add_element(element)

    # Move to new group
    scene.move_elements_to_group(elements, new_group)

    # Verify new relationships
    for element in elements:
        assert element._parent == new_group
        assert element in new_group._elements
    assert new_group in scene._elements


def test_group_registration_edge_cases():
    """Test edge cases in group registration."""
    scene = Scene()
    group = SceneGroup(id="test")
    element = Mock(spec=SceneElement)

    # Test adding unregistered group
    with pytest.raises(AssertionError):
        scene.move_elements_to_group([element], group)
    assert group in scene._elements

    scene.add_element(element, parent=group)

    with pytest.raises(AssertionError):
        scene.add_element(group)
    assert len(list(scene)) == 2
