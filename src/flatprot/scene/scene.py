# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Iterator, Generator, Set, Tuple, Dict

from flatprot.core import (
    Structure,
    logger,
)

from .base_element import BaseSceneElement
from .structure import BaseStructureSceneElement

from .group import SceneGroup
from .resolver import CoordinateResolver

# Import custom errors
from .errors import (
    SceneError,
    ElementNotFoundError,
    DuplicateElementError,
    ParentNotFoundError,
    ElementTypeError,
    CircularDependencyError,
    SceneGraphInconsistencyError,
    InvalidSceneOperationError,
)


class Scene:
    """Manages the scene graph for a protein structure visualization.

    The Scene holds the core protein structure data and a hierarchical
    tree of SceneElements (nodes), including SceneGroups.
    It provides methods to build and manipulate this tree, and to query
    elements based on residue information.
    """

    def __init__(self, structure: Structure):
        """Initializes the Scene with a core Structure object.

        Args:
            structure: The core biological structure data.
        """
        if not isinstance(structure, Structure):
            raise TypeError("Scene must be initialized with a Structure object.")

        self._structure: Structure = structure
        # List of top-level nodes (elements with no parent)
        # The order here determines the base rendering order of top-level items.
        self._nodes: List[BaseSceneElement] = []
        # For quick lookups by ID
        self._element_registry: Dict[str, BaseSceneElement] = {}
        # Create the coordinate resolver instance
        self._resolver: Optional[CoordinateResolver] = None  # Initialize as None

    @property
    def structure(self) -> Structure:
        """Get the core Structure object associated with this scene."""
        return self._structure

    @property
    def top_level_nodes(self) -> List[BaseSceneElement]:
        """Get the list of top-level nodes in the scene graph."""
        return self._nodes

    def get_element_by_id(self, id: str) -> Optional[BaseSceneElement]:
        """Retrieve a scene element by its unique ID.

        Args:
            id: The ID of the element to find.

        Returns:
            The found BaseSceneElement or None if no element has that ID.
        """
        return self._element_registry.get(id)

    def _register_element(self, element: BaseSceneElement) -> None:
        """Internal method to add an element to the ID registry."""
        if element.id in self._element_registry:
            # Raise specific error for duplicate IDs
            raise DuplicateElementError(
                f"Element with ID '{element.id}' already exists in the registry."
            )
        self._element_registry[element.id] = element

    def _unregister_element(self, element: BaseSceneElement) -> None:
        """Internal method to remove an element from the ID registry."""
        if element.id in self._element_registry:
            del self._element_registry[element.id]
        # Invalidate resolver cache if element affecting it is removed
        self._resolver = None

    def add_element(
        self, element: BaseSceneElement, parent_id: Optional[str] = None
    ) -> None:
        """Adds a SceneElement to the scene graph.

        If parent_id is provided, the element is added as a child of the
        specified parent group. Otherwise, it's added as a top-level node.

        Args:
            element: The SceneElement to add.
            parent_id: The ID of the parent SceneGroup, or None for top-level.

        Raises:
            ValueError: If parent_id is specified but not found, or if the
                        target parent is not a SceneGroup, or if the element ID
                        already exists.
            TypeError: If the element is not a BaseSceneElement.
        """
        if not isinstance(element, BaseSceneElement):
            # Raise specific type error
            raise ElementTypeError(
                f"Object to add is not a BaseSceneElement subclass (got {type(element).__name__})."
            )

        # Check for existing element using ID registry (more reliable)
        if element.id in self._element_registry:
            raise DuplicateElementError(
                f"Element with ID '{element.id}' already exists in the registry."
            )

        # Check if element object seems already attached (should have parent=None)
        # This prevents adding the same *object* instance twice if somehow unregistered but still linked
        if element.parent is not None:
            raise InvalidSceneOperationError(
                f"Element '{element.id}' already has a parent ('{element.parent.id}') and cannot be added directly."
            )

        # Check if it's already a top-level node (should not happen if parent is None check passes, but belt-and-suspenders)
        if element in self._nodes:
            raise InvalidSceneOperationError(
                f"Element '{element.id}' is already a top-level node."
            )

        self._register_element(element)  # Register first

        parent: Optional[SceneGroup] = None
        if parent_id is not None:
            potential_parent = self.get_element_by_id(parent_id)
            if potential_parent is None:
                self._unregister_element(element)  # Rollback
                # Raise specific error for parent not found
                raise ParentNotFoundError(
                    f"Parent group with ID '{parent_id}' not found."
                )
            if not isinstance(potential_parent, SceneGroup):
                self._unregister_element(element)  # Rollback
                # Raise specific type error for parent
                raise ElementTypeError(
                    f"Specified parent '{parent_id}' is not a SceneGroup (got {type(potential_parent).__name__})."
                )
            parent = potential_parent

        try:
            if parent:
                # Let add_child raise its specific errors (e.g., ValueError for circular)
                parent.add_child(element)
            else:
                element._set_parent(None)
                self._nodes.append(element)
        except (ValueError, TypeError, SceneError) as e:
            # Catch potential errors from add_child or _set_parent
            self._unregister_element(
                element
            )  # Rollback registration (also invalidates resolver)
            if parent is None and element in self._nodes:
                self._nodes.remove(element)  # Rollback adding to top-level
            # Re-raise the original specific error, don't wrap
            raise e
        # Invalidate resolver cache if element affecting it is added
        self._resolver = None

    def remove_element(self, element_id: str) -> None:
        """Removes a SceneElement and its descendants from the scene graph by ID.

        Args:
            element_id: The ID of the SceneElement to remove.

        Raises:
            ValueError: If the element with the given ID is not found in the scene.
        """
        element = self.get_element_by_id(element_id)
        if element is None:
            # Raise specific error
            raise ElementNotFoundError(f"Element with ID '{element_id}' not found.")

        # --- Collect nodes to remove --- (No change needed here)
        nodes_to_unregister: List[BaseSceneElement] = []
        nodes_to_process: List[BaseSceneElement] = [element]

        while nodes_to_process:
            node = nodes_to_process.pop(0)
            # Check if already unregistered (in case of complex graph manipulations, though ideally not needed)
            if node.id not in self._element_registry:
                continue

            nodes_to_unregister.append(node)
            if isinstance(node, SceneGroup):
                # Add children to process queue (create copy for safe iteration)
                nodes_to_process.extend(list(node.children))

        for node in nodes_to_unregister:
            self._unregister_element(node)

        # --- Detach the root element --- #
        parent = element.parent
        element_was_top_level = element in self._nodes

        if parent:
            if parent.id in self._element_registry and isinstance(parent, SceneGroup):
                try:
                    parent.remove_child(element)
                except ValueError:
                    # This *shouldn't* happen if graph is consistent. Treat as inconsistency.
                    # Log it, but also raise a specific error.
                    element._set_parent(None)
                    # Raise inconsistency error instead of just warning
                    raise SceneGraphInconsistencyError(
                        f"SceneGraph Inconsistency: Element '{element.id}' not found in supposed parent '{parent.id}' children list during removal."
                    )
            else:
                # Parent reference exists but parent is invalid/unregistered.
                element._set_parent(None)
                # This is also an inconsistency
                raise SceneGraphInconsistencyError(
                    f"SceneGraph Inconsistency: Parent '{parent.id if parent else 'None'}' of element '{element.id}' is invalid or unregistered during removal."
                )

        elif element_was_top_level:
            # If it was supposed to be top-level, remove it
            self._nodes.remove(element)
            element._set_parent(None)
        else:
            # Element was registered, had no parent, but wasn't in top-level nodes.
            # This indicates an inconsistency.
            element._set_parent(None)
            raise SceneGraphInconsistencyError(
                f"SceneGraph Inconsistency: Element '{element.id}' was registered but not found in the scene graph structure (neither parented nor top-level)."
            )
        # Invalidate resolver cache since elements were removed
        self._resolver = None

    def move_element(
        self, element_id: str, new_parent_id: Optional[str] = None
    ) -> None:
        """Moves a SceneElement identified by its ID to a new parent.

        Args:
            element_id: The ID of the SceneElement to move.
            new_parent_id: The ID of the new parent SceneGroup, or None to move
                           to the top level.

        Raises:
            ValueError: If the element or new parent is not found, if the new
                        parent is not a SceneGroup, or if the move would create
                        a circular dependency.
            TypeError: If the target parent is not a SceneGroup.
        """
        element = self.get_element_by_id(element_id)
        if element is None:
            raise ElementNotFoundError(f"Element with ID '{element_id}' not found.")

        current_parent = element.parent
        new_parent: Optional[SceneGroup] = None

        if new_parent_id is not None:
            potential_parent = self.get_element_by_id(new_parent_id)
            if potential_parent is None:
                raise ParentNotFoundError(
                    f"New parent group with ID '{new_parent_id}' not found."
                )
            if not isinstance(potential_parent, SceneGroup):
                raise ElementTypeError(
                    f"Target parent '{new_parent_id}' is not a SceneGroup (got {type(potential_parent).__name__})."
                )
            new_parent = potential_parent

            # Prevent circular dependency
            temp_check: Optional[BaseSceneElement] = new_parent
            while temp_check is not None:
                if temp_check is element:
                    raise CircularDependencyError(
                        f"Cannot move element '{element_id}' under '{new_parent_id}' - would create circular dependency."
                    )
                temp_check = temp_check.parent

        if current_parent is new_parent:
            return

        # --- Detach Phase --- #
        element_was_in_nodes = element in self._nodes
        try:
            if current_parent:
                current_parent.remove_child(
                    element
                )  # Let SceneGroup handle internal parent update
            elif element_was_in_nodes:
                self._nodes.remove(element)
                element._set_parent(None)
            elif (
                element.parent is None
            ):  # Already detached, potentially inconsistent state
                raise SceneGraphInconsistencyError(
                    f"SceneGraph Inconsistency: Element '{element_id}' was already detached before move operation."
                )
            else:  # Should not be reachable if graph is consistent
                raise SceneGraphInconsistencyError(
                    f"SceneGraph Inconsistency: Element '{element_id}' in inconsistent state during detach phase of move."
                )
        except ValueError as e:
            # If remove_child fails unexpectedly (e.g., element not found when it should be)
            element._set_parent(None)  # Force detachment
            raise SceneGraphInconsistencyError(
                "Scene Graph Inconsistency: "
                + f"Error detaching '{element_id}' from current parent '{current_parent.id if current_parent else 'None'}': {e}"
            ) from e  # Raise inconsistency

        # --- Attach Phase --- #
        try:
            if new_parent:
                new_parent.add_child(
                    element
                )  # Let add_child handle parent update & checks
            else:
                element._set_parent(None)
                self._nodes.append(element)
        except (ValueError, TypeError, SceneError) as e:
            # If attaching fails, attempt rollback (reattach to original parent/location)
            rollback_msg = f"Failed to attach element '{element_id}' to new parent '{new_parent_id}': {e}. Attempting rollback."
            try:
                if current_parent:
                    current_parent.add_child(
                        element
                    )  # Try adding back to original parent
                elif element_was_in_nodes:  # If it was originally top-level
                    element._set_parent(None)
                    self._nodes.append(element)
                # If originally detached, leave it detached
            except Exception as rollback_e:
                # Rollback failed, graph is likely inconsistent
                msg = f"Rollback failed after attach error for element '{element.id}'. Scene graph may be inconsistent. Rollback error: {rollback_e}"
                raise SceneGraphInconsistencyError(
                    msg
                ) from e  # Raise inconsistency, chaining original error

            raise InvalidSceneOperationError(rollback_msg) from e
        # Invalidate resolver cache since element position changed
        self._resolver = None

    def traverse(self) -> Generator[Tuple[BaseSceneElement, int], None, None]:
        """Performs a depth-first traversal of the scene graph.

        Yields:
            Tuple[BaseSceneElement, int]: A tuple containing the scene element
                                          and its depth in the tree (0 for top-level).
        """
        nodes_to_visit: List[Tuple[BaseSceneElement, int]] = [
            (node, 0) for node in reversed(self._nodes)
        ]
        visited_ids: Set[str] = set()

        while nodes_to_visit:
            element, depth = nodes_to_visit.pop()

            # Check registry in case node was removed during traversal (unlikely but possible)
            if element.id not in self._element_registry or element.id in visited_ids:
                continue
            visited_ids.add(element.id)

            yield element, depth

            if isinstance(element, SceneGroup):
                # Add children to the stack in reverse order to maintain visit order
                # Ensure children are also still registered before adding
                children_to_add = [
                    (child, depth + 1)
                    for child in reversed(element.children)
                    if child.id in self._element_registry
                ]
                nodes_to_visit.extend(children_to_add)

    def get_all_elements(self) -> List[BaseSceneElement]:
        """Returns a flat list of all elements in the scene graph.

        Returns:
            A list containing all BaseSceneElement objects registered in the scene.
        """
        return list(self._element_registry.values())

    def get_sequential_structure_elements(self) -> List[BaseStructureSceneElement]:
        """
        Returns a list of all BaseStructureSceneElement instances in the scene,
        sorted sequentially by chain ID and then by starting residue index.

        Assumes each BaseStructureSceneElement primarily represents a single
        contiguous range for sorting purposes.

        Returns:
            List[BaseStructureSceneElement]: Sorted list of structure elements.
        """
        structure_elements: List[BaseStructureSceneElement] = []
        for element in self._element_registry.values():
            if isinstance(element, BaseStructureSceneElement):
                structure_elements.append(element)

        def sort_key(element: BaseStructureSceneElement) -> Tuple[str, int]:
            primary_chain = "~"  # Use ~ to sort after standard chain IDs
            start_residue = float("inf")  # Sort elements without range last

            if element.residue_range_set and element.residue_range_set.ranges:
                # Use the first range (min start residue) for sorting key
                try:
                    first_range = min(
                        element.residue_range_set.ranges,
                        key=lambda r: (r.chain_id, r.start),
                    )
                    primary_chain = first_range.chain_id
                    start_residue = first_range.start
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not determine sort key for element {element.id} due to range issue: {e}"
                    )
            else:
                logger.debug(
                    f"Structure element {element.id} has no residue range for sorting."
                )

            # Return a tuple for multi-level sorting
            return (primary_chain, start_residue)

        try:
            structure_elements.sort(key=sort_key)
        except Exception as e:
            logger.error(f"Error sorting structure elements: {e}", exc_info=True)
            # Return unsorted list in case of unexpected sorting error

        return structure_elements

    def __iter__(self) -> Iterator[BaseSceneElement]:
        """Iterate over the top-level nodes of the scene graph."""
        return iter(self._nodes)

    def __len__(self) -> int:
        """Return the total number of elements currently registered in the scene."""
        return len(self._element_registry)

    def __repr__(self) -> str:
        """Provide a string representation of the scene."""
        structure_id = getattr(self.structure, "id", "N/A")  # Safely get structure ID
        return (
            f"<Scene structure_id='{structure_id}' "
            f"top_level_nodes={len(self._nodes)} total_elements={len(self)}>"
        )

    # Lazily create resolver only when needed to avoid issues during Scene init
    @property
    def resolver(self) -> CoordinateResolver:
        """Get the CoordinateResolver instance for this scene."""
        if self._resolver is None:
            # Pass the current element registry
            self._resolver = CoordinateResolver(self._structure, self._element_registry)
        return self._resolver
