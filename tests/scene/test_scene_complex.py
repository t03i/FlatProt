# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


from flatprot.scene import Scene, DuplicateElementError

# Assuming create_mock_element and fixtures are available
# If not, copy them from test_scene_system.py or create a shared conftest.py
from .test_scene_system import create_mock_element

# ruff: noqa: F401
from .test_scene_system import scene, mock_structure

# --- Complex Hierarchy Tests ---


# ruff: noqa: F811
def test_deep_nesting_add_remove(scene: Scene):
    """Test adding and removing elements in a deeply nested structure."""
    # Create: root -> g1 -> g2 -> g3 -> element
    root = create_mock_element("root", is_group=True)
    g1 = create_mock_element("g1", is_group=True)
    g2 = create_mock_element("g2", is_group=True)
    g3 = create_mock_element("g3", is_group=True)
    element = create_mock_element("deep_element")

    # Use try/except for DuplicateElementError if tests run in shared context
    try:
        scene.add_element(root)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(g1, parent_id="root")
    except DuplicateElementError:
        pass
    try:
        scene.add_element(g2, parent_id="g1")
    except DuplicateElementError:
        pass
    try:
        scene.add_element(g3, parent_id="g2")
    except DuplicateElementError:
        pass
    try:
        scene.add_element(element, parent_id="g3")
    except DuplicateElementError:
        pass

    # Assert initial state (assuming setup worked or passed)
    assert scene.get_element_by_id("deep_element") is element
    assert element.parent is g3
    assert g3.parent is g2
    assert g2.parent is g1
    assert g1.parent is root
    assert root.parent is None
    initial_len = len(scene)  # Store length after setup

    # Remove the middle group g2
    scene.remove_element("g2")

    assert len(scene) == initial_len - 3  # Should remove g2, g3, element
    assert scene.get_element_by_id("g2") is None
    assert scene.get_element_by_id("g3") is None
    assert scene.get_element_by_id("deep_element") is None
    # Check parent relationship still okay - Need to ensure root exists
    if scene.get_element_by_id("root") and scene.get_element_by_id("g1"):
        assert g1 in root.children

    # Check traversal order after removal
    expected = [(root, 0), (g1, 1)]
    result = list(scene.traverse())
    result_ids_depths = [(el.id, depth) for el, depth in result]
    expected_ids_depths = [(el.id, depth) for el, depth in expected]
    assert result_ids_depths == expected_ids_depths


# ruff: noqa: F811
def test_multiple_moves_reparenting(scene: Scene):
    """Test moving elements multiple times between different parents."""
    g1 = create_mock_element("g1_m", is_group=True)  # Use unique IDs for this test
    g2 = create_mock_element("g2_m", is_group=True)
    g3 = create_mock_element("g3_m", is_group=True)
    e1 = create_mock_element("e1_m")
    e2 = create_mock_element("e2_m")

    # Use try/except for DuplicateElementError
    try:
        scene.add_element(g1)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(g2)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(g3, parent_id=g2.id)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(e1, parent_id=g1.id)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(e2, parent_id=g1.id)
    except DuplicateElementError:
        pass

    # Assert initial state
    assert e1.parent is g1
    assert e2.parent is g1
    assert g3.parent is g2
    initial_len = len(scene)

    # 1. Move e1 from g1 to g2
    scene.move_element(e1.id, new_parent_id=g2.id)
    assert e1.parent is g2
    assert e1 not in g1.children
    assert e1 in g2.children
    assert e2.parent is g1
    assert g3.parent is g2

    # 2. Move g3 from g2 to g1
    scene.move_element(g3.id, new_parent_id=g1.id)
    assert g3.parent is g1
    assert g3 not in g2.children
    assert g3 in g1.children
    assert e1.parent is g2

    # 3. Move e2 from g1 to top level
    scene.move_element(e2.id, new_parent_id=None)
    assert e2.parent is None
    assert e2 not in g1.children
    assert e2 in scene.top_level_nodes
    assert g3.parent is g1

    # 4. Move g2 from top level to g1
    scene.move_element(g2.id, new_parent_id=g1.id)
    assert g2.parent is g1
    assert g2 not in scene.top_level_nodes
    assert g2 in g1.children
    assert e1.parent is g2  # e1 is child of g2, moves with it
    assert g3.parent is g1

    # Check final structure via traversal
    # Expected order: g1, g3, g2, e1, e2 (top-level nodes: g1, e2)
    expected_final = [
        (g1, 0),
        (g3, 1),
        (g2, 1),
        (e1, 2),
        (e2, 0),
    ]
    result = list(scene.traverse())
    result_ids_depths = [(el.id, depth) for el, depth in result]
    expected_ids_depths = [(el.id, depth) for el, depth in expected_final]
    assert result_ids_depths == expected_ids_depths
    assert len(scene) == initial_len  # No elements lost


# ruff: noqa: F811
def test_interleaved_add_move_remove(scene: Scene):
    """Test a sequence of adds, moves, and removes."""
    # Use unique IDs for this test
    r = create_mock_element("r_i", is_group=True)
    a = create_mock_element("a_i", is_group=True)
    b = create_mock_element("b_i")
    c = create_mock_element("c_i")
    d = create_mock_element("d_i")

    # 1. Initial setup: r -> a -> b, c (top-level), d (top-level)
    # Use try/except for DuplicateElementError
    try:
        scene.add_element(r)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(a, parent_id=r.id)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(b, parent_id=a.id)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(c)
    except DuplicateElementError:
        pass
    try:
        scene.add_element(d)
    except DuplicateElementError:
        pass

    initial_len = len(scene)
    _ = sorted([n.id for n in scene.top_level_nodes])

    # 2. Move c under a
    scene.move_element(c.id, new_parent_id=a.id)
    assert c.parent is a
    assert c not in scene.top_level_nodes
    assert sorted([child.id for child in a.children]) == [b.id, c.id]

    # 3. Remove b
    scene.remove_element(b.id)
    assert scene.get_element_by_id(b.id) is None
    assert b not in a.children
    assert len(a.children) == 1
    assert len(scene) == initial_len - 1

    # 4. Move a to top level
    scene.move_element(a.id, new_parent_id=None)
    assert a.parent is None
    assert a in scene.top_level_nodes
    assert a not in r.children
    assert c.parent is a  # c moves with a
    # Top level should now include a, keep d, keep r
    current_top_level_ids = sorted([n.id for n in scene.top_level_nodes])
    assert a.id in current_top_level_ids
    assert d.id in current_top_level_ids
    assert r.id in current_top_level_ids

    # 5. Add new element e under r
    e = create_mock_element("e_i")
    try:
        scene.add_element(e, parent_id=r.id)
    except DuplicateElementError:
        pass

    assert e.parent is r
    assert e in r.children
    assert len(scene) == initial_len  # (initial - removed b + added e)

    # 6. Remove r (should remove e too)
    scene.remove_element(r.id)
    assert scene.get_element_by_id(r.id) is None
    assert scene.get_element_by_id(e.id) is None
    assert len(scene) == initial_len - 2  # Removed r, e from state after step 5
    # Top level should now be only a and d
    final_top_level_ids = sorted([n.id for n in scene.top_level_nodes])
    assert final_top_level_ids == sorted([a.id, d.id])
    assert c.parent is a  # Verify c is still under a
