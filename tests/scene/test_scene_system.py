import pytest
from unittest.mock import Mock, MagicMock  # Use unittest.mock

from flatprot.scene import (
    Scene,
    SceneGroup,
    BaseSceneElement,
    DuplicateElementError,
    ElementNotFoundError,
    ParentNotFoundError,
    ElementTypeError,
    CircularDependencyError,
    SceneGraphInconsistencyError,
    BaseStructureSceneElement,
)
from flatprot.core import Structure, ResidueRangeSet


# Helper function to create a mock element
def create_mock_element(element_id: str, is_group: bool = False) -> Mock | MagicMock:
    """Creates a mock BaseSceneElement or SceneGroup."""
    if is_group:
        # SceneGroup needs children list, add_child, remove_child
        mock = MagicMock(spec=SceneGroup)
        mock.children = []
        # Initialize internal parent state for the group itself
        mock._parent = None

        def add_child_side_effect(child):
            if not isinstance(child, BaseSceneElement):
                raise ElementTypeError("Not a BaseSceneElement")
            if child is mock:
                raise ValueError("Cannot add self as child")
            # Simplified circular check for mock using internal parent state
            temp = mock._parent
            while temp:
                if temp is child:
                    raise CircularDependencyError("Circular dependency detected")
                temp = temp._parent  # Traverse internal parent
            mock.children.append(child)
            # Simulate the child's parent being set by calling its mock _set_parent
            child._set_parent(mock)

        def remove_child_side_effect(child):
            if child not in mock.children:
                raise ValueError("Child not found")
            mock.children.remove(child)
            # Simulate the child's parent being unset by calling its mock _set_parent
            child._set_parent(None)

        mock.add_child.side_effect = add_child_side_effect
        mock.remove_child.side_effect = remove_child_side_effect
    else:
        mock = MagicMock(spec=BaseSceneElement)
        mock._parent = None  # Internal state

    mock.id = element_id
    # Make the 'parent' property dynamically read the internal '_parent' state
    type(mock).parent = property(fget=lambda self_mock: self_mock._parent)
    mock.residue_range_set = None  # Initialize

    # Mock the _set_parent method to update the internal state
    def set_parent_side_effect(parent):
        mock._parent = parent

    mock._set_parent = MagicMock(side_effect=set_parent_side_effect)

    return mock


@pytest.fixture
def mock_structure() -> Mock:
    """Fixture for a mock Structure object."""
    return Mock(spec=Structure)


@pytest.fixture
def scene(mock_structure: Mock) -> Scene:
    """Fixture for a Scene instance initialized with a mock Structure."""
    return Scene(structure=mock_structure)


# --------------------------
# 1. Initialization & Properties
# --------------------------


def test_scene_initialization(mock_structure: Mock) -> None:
    """Test scene initialization and basic properties."""
    scene = Scene(structure=mock_structure)
    assert scene.structure is mock_structure
    assert scene.top_level_nodes == []
    assert len(scene) == 0
    assert (
        repr(scene) == "<Scene structure_id='N/A' top_level_nodes=0 total_elements=0>"
    )


def test_scene_initialization_wrong_type() -> None:
    """Test that initializing Scene with non-Structure raises TypeError."""
    with pytest.raises(
        TypeError, match="Scene must be initialized with a Structure object"
    ):
        Scene(structure="not a structure")  # type: ignore


# --------------------------
# 2. Node Addition Tests (`add_node`)
# --------------------------


def test_add_top_level_node(scene: Scene) -> None:
    """Test adding a single node to the top level."""
    element = create_mock_element("elem1")
    scene.add_element(element)

    assert element in scene.top_level_nodes
    assert scene.get_element_by_id("elem1") is element
    assert element.parent is None
    element._set_parent.assert_called_once_with(None)
    assert len(scene) == 1


def test_add_child_node(scene: Scene) -> None:
    """Test adding a node as a child of an existing group."""
    parent_group = create_mock_element("group1", is_group=True)
    child_element = create_mock_element("child1")

    scene.add_element(parent_group)
    scene.add_element(child_element, parent_id="group1")

    assert child_element not in scene.top_level_nodes
    assert scene.get_element_by_id("child1") is child_element
    assert child_element.parent is parent_group  # Should work now
    parent_group.add_child.assert_called_once_with(child_element)
    # _set_parent is called by the mock group's add_child side effect
    child_element._set_parent.assert_called_once_with(parent_group)
    assert len(scene) == 2


def test_add_node_duplicate_id(scene: Scene) -> None:
    """Test adding a node with an ID that already exists."""
    element1 = create_mock_element("elem1")
    element2 = create_mock_element("elem1")  # Same ID
    scene.add_element(element1)

    with pytest.raises(
        DuplicateElementError, match="Element with ID 'elem1' already exists"
    ):
        scene.add_element(element2)
    assert len(scene) == 1


def test_add_node_parent_not_found(scene: Scene) -> None:
    """Test adding a node with a non-existent parent ID."""
    element = create_mock_element("elem1")
    with pytest.raises(
        ParentNotFoundError, match="Parent group with ID 'nonexistent' not found"
    ):
        scene.add_element(element, parent_id="nonexistent")
    assert len(scene) == 0


def test_add_node_parent_not_group(scene: Scene) -> None:
    """Test adding a node with a parent that is not a SceneGroup."""
    parent_element = create_mock_element("parent1", is_group=False)
    child_element = create_mock_element("child1")
    scene.add_element(parent_element)

    with pytest.raises(
        ElementTypeError, match="Specified parent 'parent1' is not a SceneGroup"
    ):
        scene.add_element(child_element, parent_id="parent1")
    assert len(scene) == 1


def test_add_node_invalid_type(scene: Scene) -> None:
    """Test adding an object that is not a BaseSceneElement."""
    with pytest.raises(
        ElementTypeError, match="Object to add is not a BaseSceneElement subclass"
    ):
        scene.add_element("not_an_element")  # type: ignore


def test_add_node_already_parented(scene: Scene) -> None:
    """Test adding a node that already exists raises DuplicateElementError first."""
    group = create_mock_element("group1", is_group=True)
    element = create_mock_element("elem1")
    scene.add_element(group)
    scene.add_element(element, parent_id="group1")  # elem1 is now child of group1
    assert element.parent is group  # Verify parenting worked

    # Try adding the same element object again - DuplicateElementError is raised first
    with pytest.raises(
        DuplicateElementError, match="Element with ID 'elem1' already exists"
    ):
        scene.add_element(element)  # Try adding as top-level

    with pytest.raises(
        DuplicateElementError, match="Element with ID 'elem1' already exists"
    ):
        scene.add_element(
            element, parent_id="group1"
        )  # Try adding to same parent again


# --------------------------
# 3. Node Removal Tests (`remove_node`)
# --------------------------


def test_remove_top_level_node(scene: Scene) -> None:
    """Test removing a top-level node."""
    element = create_mock_element("elem1")
    scene.add_element(element)
    assert scene.get_element_by_id("elem1") is element
    assert element.parent is None  # Verify initial state

    scene.remove_element("elem1")

    assert element not in scene.top_level_nodes
    assert scene.get_element_by_id("elem1") is None
    assert element.parent is None  # Should be detached
    # Check if _set_parent was called during removal (it should be)
    # It's called once on add (None) and once on remove (None)
    assert (
        element._set_parent.call_count == 2
    )  # Once for add(None), once for remove(None)
    element._set_parent.assert_called_with(None)  # Last call

    assert len(scene) == 0


def test_remove_child_node(scene: Scene) -> None:
    """Test removing a child node."""
    parent_group = create_mock_element("group1", is_group=True)
    child_element = create_mock_element("child1")
    scene.add_element(parent_group)
    scene.add_element(child_element, parent_id="group1")
    assert child_element.parent is parent_group  # Verify initial state

    scene.remove_element("child1")

    assert scene.get_element_by_id("child1") is None
    assert child_element not in parent_group.children
    parent_group.remove_child.assert_called_once_with(child_element)
    # _set_parent is called by the mock group's remove_child side effect
    assert (
        child_element._set_parent.call_count == 2
    )  # Once for add(parent), once for remove(None)
    child_element._set_parent.assert_called_with(None)  # Last call
    assert child_element.parent is None  # Verify detached state
    assert len(scene) == 1  # Parent group remains


def test_remove_group_with_children(scene: Scene) -> None:
    """Test removing a group node also removes its descendants."""
    group = create_mock_element("group1", is_group=True)
    child1 = create_mock_element("child1")
    child2 = create_mock_element("child2")
    # Need a sub-group to add grandchild
    sub_group = create_mock_element("subgroup", is_group=True)
    grandchild = create_mock_element("grandchild1")

    scene.add_element(group)
    scene.add_element(child1, parent_id="group1")
    scene.add_element(sub_group, parent_id="group1")
    scene.add_element(
        child2, parent_id="group1"
    )  # Add child2 after subgroup for ordering check if needed
    scene.add_element(grandchild, parent_id="subgroup")

    assert len(scene) == 5  # group1, child1, subgroup, child2, grandchild1
    assert grandchild.parent is sub_group
    assert sub_group.parent is group
    assert child1.parent is group
    assert child2.parent is group

    # Store references before removal for later checks if needed
    # all_elements = [group, child1, sub_group, child2, grandchild]

    scene.remove_element("group1")

    assert scene.get_element_by_id("group1") is None
    assert scene.get_element_by_id("child1") is None
    assert scene.get_element_by_id("child2") is None
    assert scene.get_element_by_id("subgroup") is None
    assert scene.get_element_by_id("grandchild1") is None
    assert len(scene) == 0

    # The root element being removed should have its parent reset.
    # Descendants are unregistered but their mock parent refs might persist internally.
    assert group.parent is None
    group._set_parent.assert_called_with(None)  # Check it was called during removal


def test_remove_node_not_found(scene: Scene) -> None:
    """Test removing a node with an ID that does not exist."""
    with pytest.raises(
        ElementNotFoundError, match="Element with ID 'nonexistent' not found"
    ):
        scene.remove_element("nonexistent")


def test_remove_node_inconsistency_parent_mismatch(scene: Scene) -> None:
    """Test removing a node whose parent doesn't list it as a child (inconsistency)."""
    parent_group = create_mock_element("group1", is_group=True)
    child_element = create_mock_element("child1")

    scene.add_element(parent_group)
    scene.add_element(child_element, parent_id="group1")
    assert child_element.parent is parent_group  # Verify parent is set

    # Manually break consistency: remove child from parent's list but keep parent pointer
    # and mock remove_child to fail as if the child isn't there.
    parent_group.children.remove(child_element)
    parent_group.remove_child.side_effect = ValueError("Child not found")

    # Removal attempt should trigger the `except ValueError` block in `remove_node`
    expected_error_msg = r"SceneGraph Inconsistency: Element 'child1' not found in supposed parent 'group1' children list during removal."
    with pytest.raises(SceneGraphInconsistencyError, match=expected_error_msg):
        scene.remove_element("child1")

    # Check state after attempted removal
    assert scene.get_element_by_id("child1") is None  # Should still be unregistered
    assert child_element.parent is None  # Should be detached by the error handler
    child_element._set_parent.assert_called_with(None)  # Verify detachment call
    assert len(scene) == 1  # Parent remains


def test_remove_node_inconsistency_unregistered_parent(scene: Scene) -> None:
    """Test removing node when parent ref exists but parent is unregistered."""
    parent_group = create_mock_element("group1", is_group=True)
    child_element = create_mock_element("child1")

    scene.add_element(parent_group)
    scene.add_element(child_element, parent_id="group1")
    assert child_element.parent is parent_group

    # Manually break consistency: unregister the parent
    del scene._element_registry[parent_group.id]

    # Removal attempt should hit the inconsistency check for invalid parent
    expected_error_msg = r"SceneGraph Inconsistency: Parent 'group1' of element 'child1' is invalid or unregistered during removal."
    with pytest.raises(SceneGraphInconsistencyError, match=expected_error_msg):
        scene.remove_element("child1")

    # Check state after attempted removal
    assert scene.get_element_by_id("child1") is None  # Child should be unregistered
    assert child_element.parent is None  # Child should be detached
    assert len(scene) == 0  # Only parent was left, which we manually removed


def test_remove_node_inconsistency_orphaned_element(scene: Scene) -> None:
    """Test removing node that has no parent but isn't in top-level nodes."""
    element = create_mock_element("elem1")

    scene.add_element(element)
    assert element in scene.top_level_nodes
    assert element.parent is None

    # Manually break consistency: remove from top-level list without unregistering/detaching
    scene._nodes.remove(element)

    # Removal attempt should hit the final inconsistency check
    expected_error_msg = r"SceneGraph Inconsistency: Element 'elem1' was registered but not found in the scene graph structure \(neither parented nor top-level\)."
    with pytest.raises(SceneGraphInconsistencyError, match=expected_error_msg):
        scene.remove_element("elem1")

    # Check state after attempted removal
    assert scene.get_element_by_id("elem1") is None  # Element should be unregistered
    assert element.parent is None  # Element should be detached
    assert len(scene) == 0


# --------------------------
# 4. Node Moving Tests (`move_node`)
# --------------------------


def test_move_node_to_new_parent(scene: Scene) -> None:
    """Test moving a node from one parent to another."""
    group1 = create_mock_element("group1", is_group=True)
    group2 = create_mock_element("group2", is_group=True)
    element = create_mock_element("elem1")

    scene.add_element(group1)
    scene.add_element(group2)
    scene.add_element(element, parent_id="group1")

    assert element.parent is group1  # Verify initial state
    assert element in group1.children
    assert element not in group2.children

    scene.move_element("elem1", new_parent_id="group2")

    assert element.parent is group2  # Verify new parent
    assert element not in group1.children
    assert element in group2.children
    group1.remove_child.assert_called_once_with(element)
    group2.add_child.assert_called_once_with(element)
    assert element not in scene.top_level_nodes
    assert len(scene) == 3

    # Check _set_parent calls: add(g1), remove(None triggered by g1.remove_child), add(g2 triggered by g2.add_child)
    assert element._set_parent.call_count == 3
    element._set_parent.assert_called_with(group2)  # Last call


def test_move_node_to_top_level(scene: Scene) -> None:
    """Test moving a node from a parent to the top level."""
    group1 = create_mock_element("group1", is_group=True)
    element = create_mock_element("elem1")

    scene.add_element(group1)
    scene.add_element(element, parent_id="group1")
    assert element.parent is group1  # Verify initial state

    scene.move_element("elem1", new_parent_id=None)

    assert element.parent is None  # Verify new parent (None)
    assert element not in group1.children
    group1.remove_child.assert_called_once_with(element)
    assert element in scene.top_level_nodes
    assert len(scene) == 2

    # Check _set_parent calls: add(g1), remove(None triggered by g1.remove_child), remove(None explicit in move_node)
    assert element._set_parent.call_count == 3
    element._set_parent.assert_called_with(None)  # Last call


def test_move_top_level_node_to_parent(scene: Scene) -> None:
    """Test moving a top-level node to a parent group."""
    group1 = create_mock_element("group1", is_group=True)
    element = create_mock_element("elem1")

    scene.add_element(group1)
    scene.add_element(element)  # Add as top-level
    assert element.parent is None  # Verify initial state
    assert element in scene.top_level_nodes

    scene.move_element("elem1", new_parent_id="group1")

    assert element.parent is group1  # Verify new parent
    assert element in group1.children
    group1.add_child.assert_called_once_with(element)
    assert element not in scene.top_level_nodes
    assert len(scene) == 2

    # Check _set_parent calls: add(None), remove(None explicit in move_node), add(g1 triggered by g1.add_child)
    assert element._set_parent.call_count == 3
    element._set_parent.assert_called_with(group1)  # Last call


def test_move_node_element_not_found(scene: Scene) -> None:
    """Test moving a non-existent node."""
    group1 = create_mock_element("group1", is_group=True)
    scene.add_element(group1)
    with pytest.raises(
        ElementNotFoundError, match="Element with ID 'nonexistent' not found"
    ):
        scene.move_element("nonexistent", new_parent_id="group1")


def test_move_node_parent_not_found(scene: Scene) -> None:
    """Test moving a node to a non-existent parent."""
    element = create_mock_element("elem1")
    scene.add_element(element)
    with pytest.raises(
        ParentNotFoundError, match="New parent group with ID 'nonexistent' not found"
    ):
        scene.move_element("elem1", new_parent_id="nonexistent")


def test_move_node_parent_not_group(scene: Scene) -> None:
    """Test moving a node to a parent that is not a SceneGroup."""
    parent_element = create_mock_element("parent1", is_group=False)
    element = create_mock_element("elem1")
    scene.add_element(parent_element)
    scene.add_element(element)

    with pytest.raises(
        ElementTypeError, match="Target parent 'parent1' is not a SceneGroup"
    ):
        scene.move_element("elem1", new_parent_id="parent1")


def test_move_node_circular_dependency(scene: Scene) -> None:
    """Test moving a node that would create a circular dependency."""
    group1 = create_mock_element("group1", is_group=True)
    group2 = create_mock_element("group2", is_group=True)
    scene.add_element(group1)
    scene.add_element(group2, parent_id="group1")  # group2 is child of group1
    assert group2.parent is group1  # Verify setup

    # Try moving group1 under group2 (its own descendant)
    with pytest.raises(
        CircularDependencyError, match="would create circular dependency"
    ):
        scene.move_element("group1", new_parent_id="group2")

    # Ensure state is unchanged
    assert group1.parent is None
    assert group2.parent is group1
    assert group1 in scene.top_level_nodes
    assert group2 in group1.children


def test_move_node_no_change(scene: Scene) -> None:
    """Test moving a node to its current parent (should do nothing)."""
    group1 = create_mock_element("group1", is_group=True)
    element = create_mock_element("elem1")

    # Spy on methods BEFORE adding the node
    element_set_parent_spy = MagicMock(side_effect=element._set_parent.side_effect)
    element._set_parent = element_set_parent_spy
    group1_remove_child_spy = MagicMock(side_effect=group1.remove_child.side_effect)
    group1.remove_child = group1_remove_child_spy
    group1_add_child_spy = MagicMock(side_effect=group1.add_child.side_effect)
    group1.add_child = group1_add_child_spy

    scene.add_element(group1)
    scene.add_element(element, parent_id="group1")
    assert element.parent is group1  # Verify setup
    # Verify the initial add call
    element_set_parent_spy.assert_called_once_with(group1)

    # Reset spies before the move operation we want to test
    element_set_parent_spy.reset_mock()
    group1_remove_child_spy.reset_mock()
    group1_add_child_spy.reset_mock()

    # Perform the move operation (which should do nothing)
    scene.move_element("elem1", new_parent_id="group1")

    # Assert that the spies were NOT called during the move
    group1_remove_child_spy.assert_not_called()
    group1_add_child_spy.assert_not_called()
    element_set_parent_spy.assert_not_called()
    assert element.parent is group1  # Still parented


def test_move_node_attach_and_rollback_failure(scene: Scene) -> None:
    """Test move_node raising inconsistency error when attach and rollback fail."""
    group1 = create_mock_element("group1", is_group=True)
    group2 = create_mock_element("group2", is_group=True)
    element = create_mock_element("elem1")

    scene.add_element(group1)
    scene.add_element(group2)
    scene.add_element(element, parent_id="group1")
    initial_set_parent_call_count = element._set_parent.call_count

    # Simulate attach failure
    attach_error = ValueError("Attach failed!")
    group2.add_child = MagicMock(side_effect=attach_error)

    # Simulate rollback failure (reattaching to group1)
    rollback_error = ValueError("Rollback failed!")
    group1.add_child = MagicMock(side_effect=rollback_error)

    # Perform the move that will fail attach and rollback
    expected_error_msg = r"Rollback failed after attach error for element 'elem1'. Scene graph may be inconsistent. Rollback error: Rollback failed!"
    with pytest.raises(
        SceneGraphInconsistencyError, match=expected_error_msg
    ) as excinfo:
        scene.move_element("elem1", new_parent_id="group2")

    # Check that the original attach error is chained
    assert excinfo.value.__cause__ is attach_error

    # Verify final state (should be inconsistent)
    assert scene.get_element_by_id("elem1") is element  # Still registered!
    assert element.parent is None  # Should be detached by initial remove
    assert element not in group1.children  # Removed from original parent
    assert element not in group2.children  # Attach failed
    assert element not in scene.top_level_nodes  # Rollback to top-level didn't happen
    assert len(scene) == 3  # All elements technically still registered

    # Check _set_parent calls: initial add, detach from group1 (via remove_child)
    assert element._set_parent.call_count == initial_set_parent_call_count + 1
    element._set_parent.assert_called_with(None)  # Last call was detach


# --------------------------
# 5. Traversal Tests (`traverse`)
# --------------------------


def test_traverse_empty_scene(scene: Scene) -> None:
    """Test traversing an empty scene."""
    assert list(scene.traverse()) == []


def test_traverse_simple_hierarchy(scene: Scene) -> None:
    """Test traversing a scene with a simple hierarchy."""
    g1 = create_mock_element("g1", is_group=True)
    e1 = create_mock_element("e1")
    e2 = create_mock_element("e2")
    g2 = create_mock_element("g2", is_group=True)
    e3 = create_mock_element("e3")

    # Add in specific order for predictable traversal
    scene.add_element(g1)
    scene.add_element(g2)
    scene.add_element(e1, parent_id="g1")
    scene.add_element(e2, parent_id="g1")
    scene.add_element(e3, parent_id="g2")

    # Expected DFS order: g1, e1, e2, g2, e3 (based on add order of top-level)
    expected = [
        (g1, 0),
        (e1, 1),
        (e2, 1),
        (g2, 0),
        (e3, 1),
    ]
    result = list(scene.traverse())
    # Compare IDs and depths for stable comparison with mocks
    result_ids_depths = [(el.id, depth) for el, depth in result]
    expected_ids_depths = [(el.id, depth) for el, depth in expected]
    assert result_ids_depths == expected_ids_depths


def test_traverse_deeper_hierarchy(scene: Scene) -> None:
    """Test traversing a deeper hierarchy."""
    r = create_mock_element("root", is_group=True)  # Top
    c1 = create_mock_element("child1", is_group=True)
    c2 = create_mock_element("child2")
    gc1 = create_mock_element("grandchild1")
    gc2 = create_mock_element("grandchild2", is_group=True)
    ggc1 = create_mock_element("greatgrandchild1")

    scene.add_element(r)
    scene.add_element(c1, parent_id="root")
    scene.add_element(c2, parent_id="root")
    scene.add_element(gc1, parent_id="child1")
    scene.add_element(gc2, parent_id="child1")
    scene.add_element(ggc1, parent_id="grandchild2")

    # Expected DFS order: r, c1, gc1, gc2, ggc1, c2
    expected = [
        (r, 0),
        (c1, 1),
        (gc1, 2),
        (gc2, 2),
        (ggc1, 3),
        (c2, 1),
    ]
    result = list(scene.traverse())
    # Compare IDs and depths
    result_ids_depths = [(el.id, depth) for el, depth in result]
    expected_ids_depths = [(el.id, depth) for el, depth in expected]
    assert result_ids_depths == expected_ids_depths


# --------------------------
# 7. ID Lookup Tests (`get_element_by_id`)
# --------------------------


def test_get_element_by_id_found(scene: Scene) -> None:
    """Test retrieving an element by its ID when it exists."""
    element = create_mock_element("elem1")
    scene.add_element(element)
    found = scene.get_element_by_id("elem1")
    assert found is element


def test_get_element_by_id_not_found(scene: Scene) -> None:
    """Test retrieving an element by ID when it does not exist."""
    found = scene.get_element_by_id("nonexistent")
    assert found is None


# --------------------------
# 8. Iteration & Length
# --------------------------


def test_scene_iteration(scene: Scene) -> None:
    """Test iterating over the scene yields top-level nodes."""
    e1 = create_mock_element("e1")
    g1 = create_mock_element("g1", is_group=True)
    e2 = create_mock_element("e2")  # Child

    scene.add_element(e1)
    scene.add_element(g1)
    scene.add_element(e2, parent_id="g1")

    top_level_nodes = list(scene)
    assert top_level_nodes == [e1, g1]  # Order matters based on addition
    assert e2 not in top_level_nodes


def test_scene_length(scene: Scene) -> None:
    """Test the __len__ method returns the total number of registered elements."""
    e1 = create_mock_element("e1")
    g1 = create_mock_element("g1", is_group=True)
    e2 = create_mock_element("e2")

    assert len(scene) == 0
    scene.add_element(e1)
    assert len(scene) == 1
    scene.add_element(g1)
    assert len(scene) == 2
    scene.add_element(e2, parent_id="g1")
    assert len(scene) == 3
    scene.remove_element("e1")
    assert len(scene) == 2
    scene.remove_element("g1")  # Removes g1 and e2
    assert len(scene) == 0


# -----------------------------
# 9. get_all_elements Test
# -----------------------------


def test_get_all_elements(scene: Scene) -> None:
    """Test retrieving all elements in the scene."""
    e1 = create_mock_element("e1")
    g1 = create_mock_element("g1", is_group=True)
    e2 = create_mock_element("e2")

    scene.add_element(e1)
    scene.add_element(g1)
    scene.add_element(e2, parent_id="g1")

    all_elements = scene.get_all_elements()
    assert len(all_elements) == 3
    assert e1 in all_elements
    assert g1 in all_elements
    assert e2 in all_elements

    # Check IDs to be sure
    all_ids = sorted([el.id for el in all_elements])
    assert all_ids == ["e1", "e2", "g1"]

    scene.remove_element("g1")  # Removes g1 and e2
    all_elements_after_remove = scene.get_all_elements()
    assert len(all_elements_after_remove) == 1
    assert e1 in all_elements_after_remove
    assert g1 not in all_elements_after_remove
    assert e2 not in all_elements_after_remove


# -------------------------------------
# 10. Sequential Element Retrieval Tests
# -------------------------------------


def test_get_sequential_structure_elements_empty(scene: Scene) -> None:
    """Test retrieving sequential elements from an empty scene."""
    assert scene.get_sequential_structure_elements() == []


def test_get_sequential_structure_elements_single_chain(
    scene: Scene, element_a10_20: Mock, element_a5_8: Mock
) -> None:
    """Test sorting elements on a single chain."""
    scene.add_element(element_a10_20)  # Added out of order
    scene.add_element(element_a5_8)
    sorted_elements = scene.get_sequential_structure_elements()
    assert [el.id for el in sorted_elements] == ["A_5_8", "A_10_20"]


def test_get_sequential_structure_elements_multi_chain(
    scene: Scene, element_a10_20: Mock, element_a5_8: Mock, element_b1_5: Mock
) -> None:
    """Test sorting elements across multiple chains."""
    scene.add_element(element_b1_5)  # Added out of order
    scene.add_element(element_a10_20)
    scene.add_element(element_a5_8)
    sorted_elements = scene.get_sequential_structure_elements()
    assert [el.id for el in sorted_elements] == ["A_5_8", "A_10_20", "B_1_5"]


def test_get_sequential_structure_elements_with_no_range(
    scene: Scene, element_a5_8: Mock, element_no_range: Mock, element_empty_range: Mock
) -> None:
    """Test sorting includes elements with no/empty ranges (sorted last)."""
    scene.add_element(element_no_range)
    scene.add_element(element_a5_8)
    scene.add_element(element_empty_range)

    sorted_elements = scene.get_sequential_structure_elements()
    # Elements without range should sort after those with ranges
    # Their relative order might depend on internal dict iteration order, but they come last.
    sorted_ids = [el.id for el in sorted_elements]
    assert sorted_ids[0] == "A_5_8"
    assert set(sorted_ids[1:]) == {"no_range", "empty_range"}


def test_get_sequential_structure_elements_only_structure_elements(
    scene: Scene, element_a5_8: Mock
) -> None:
    """Test that only structure elements are included, not groups or others."""
    group = create_mock_element("group1", is_group=True)
    scene.add_element(element_a5_8)
    scene.add_element(group)

    sorted_elements = scene.get_sequential_structure_elements()
    assert len(sorted_elements) == 1
    assert sorted_elements[0] is element_a5_8


# --- Fixtures for Sorting Tests ---


@pytest.fixture
def element_a10_20() -> Mock:
    el = create_mock_element("A_10_20", is_group=False)
    el.residue_range_set = ResidueRangeSet.from_string("A:10-20")
    # Ensure it's recognized as a structure element for sorting
    el.__class__ = BaseStructureSceneElement
    return el


@pytest.fixture
def element_a5_8() -> Mock:
    el = create_mock_element("A_5_8", is_group=False)
    el.residue_range_set = ResidueRangeSet.from_string("A:5-8")
    el.__class__ = BaseStructureSceneElement
    return el


@pytest.fixture
def element_b1_5() -> Mock:
    el = create_mock_element("B_1_5", is_group=False)
    el.residue_range_set = ResidueRangeSet.from_string("B:1-5")
    el.__class__ = BaseStructureSceneElement
    return el


@pytest.fixture
def element_no_range() -> Mock:
    el = create_mock_element("no_range", is_group=False)
    el.residue_range_set = None
    el.__class__ = BaseStructureSceneElement
    return el


@pytest.fixture
def element_empty_range() -> Mock:
    el = create_mock_element("empty_range", is_group=False)
    el.residue_range_set = ResidueRangeSet([])  # Empty set
    el.__class__ = BaseStructureSceneElement
    return el
