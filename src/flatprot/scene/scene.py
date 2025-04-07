# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict

from .structure import StructureSceneElement
from .elements import SceneGroup, SceneElement
from flatprot.core import ResidueCoordinate, ResidueRange, ResidueRangeSet


class Scene:
    """A container that manages structural elements and their hierarchical organization.

    The Scene class serves two main purposes:
    1. Maintains a registry of all structural elements and their residue mappings
    2. Manages a hierarchical tree of elements and groups for SVG rendering

    Key Features:
    - Tracks residue mappings between structural elements and protein chain positions
    - Provides methods to add, move, and organize elements within groups
    - Maintains parent-child relationships between elements
    - Supports querying elements by residue position

    The scene has a root group that serves as the default parent for elements
    when no specific parent is provided.
    """

    def __init__(self):
        self._elements: list[SceneElement] = []
        self._residue_mappings: Dict[StructureSceneElement, ResidueRangeSet] = {}
        self._residue_ranges: dict[str, list[tuple[range, StructureSceneElement]]] = {}
        self._root_group = SceneGroup(id="root")

    def _register_element(
        self,
        element: SceneElement,
        range_set: Optional[ResidueRangeSet] = None,
    ) -> None:
        """Internal method to register a structural element in the scene.

        Updates residue mappings if the element is a StructureSceneElement.
        """
        if range_set is not None:
            self._residue_mappings[element] = range_set
        self._elements.append(element)

    def _unregister_element(self, element: SceneElement) -> ResidueRangeSet | None:
        """Internal method to unregister a structural element from the scene."""
        range_set = None
        if element in self._residue_mappings:
            range_set = self._residue_mappings[element]
            del self._residue_mappings[element]

        self._elements.remove(element)
        return range_set

    def _set_parent(self, element: SceneElement, parent: SceneGroup) -> None:
        """Internal method to set the parent of an element."""
        assert element in self._elements, "Element must be registered in the scene"
        assert (
            parent is None or parent in self._elements
        ), "Parent group must be registered in the scene"
        assert parent is None or isinstance(
            parent, SceneGroup
        ), "Parent must be a SceneGroup instance"

        if parent is None:
            parent = self._root_group

        element._parent = parent

        if element not in parent._elements:
            parent.add_element(element)

    def _remove_parent(self, element: SceneElement) -> None:
        """Internal method to remove the parent of an element."""
        assert element in self._elements, "Element must be registered in the scene"

        if element.parent:
            element.parent.remove_element(element)

        element._parent = None

    def get_elements_for_residue(
        self, residue: ResidueCoordinate
    ) -> list[StructureSceneElement]:
        """Get all structural elements containing the specified residue."""
        matching_elements = []
        for element, range_set in self._residue_mappings.items():
            if isinstance(element, StructureSceneElement):
                for range_ in range_set.ranges:
                    if (
                        range_.chain_id == residue.chain_id
                        and range_.start <= residue.residue_index <= range_.end
                    ):
                        matching_elements.append(element)
                        break
        return matching_elements

    def get_element_index_from_residue(
        self, residue: ResidueCoordinate, element: StructureSceneElement
    ) -> int:
        """Convert a ResidueCoordinate to a 0-based index within a specific element.

        Args:
            residue: The ResidueCoordinate (chain and index) to convert.
            element: The StructureSceneElement to get the local index for.

        Returns:
            The 0-based index within the element.

        Raises:
            AssertionError: If the element is not registered or the residue is
                          not within the element's range set.
        """
        assert (
            element in self._residue_mappings
        ), "Element must be registered in the scene"
        range_set = self._residue_mappings[element]
        # Check if the specific ResidueCoordinate is within the element's ranges
        assert (
            residue in range_set
        ), f"Residue {residue} must be in the element's residue range set {range_set}"

        # Find the specific range the residue belongs to (assuming non-overlapping ranges within an element)
        # and calculate the offset. This assumes simple continuous ranges for now.
        # More complex logic might be needed if elements can map discontinuous ranges.
        # For now, we'll assume the first range is the relevant one for simplicity,
        # matching the previous logic, but this might need refinement if elements
        # can represent multiple discontinuous segments.
        relevant_range = next(
            (
                r
                for r in range_set.ranges
                if r.chain_id == residue.chain_id
                and r.start <= residue.residue_index <= r.end
            ),
            None,
        )
        if relevant_range is None:
            # This should theoretically not happen due to the 'residue in range_set' check above,
            # but added for robustness.
            raise ValueError(
                f"Could not find range containing {residue} in element's range set {range_set}"
            )

        return residue.residue_index - relevant_range.start

    def get_elements_for_residue_range(
        self, residue_range: ResidueRange
    ) -> list[StructureSceneElement]:
        """Get all structural elements overlapping with the specified range."""
        matching_elements = []
        for element, range_set in self._residue_mappings.items():
            if isinstance(element, StructureSceneElement):
                for range_ in range_set.ranges:
                    if range_.chain_id == residue_range.chain_id and not (
                        range_.end < residue_range.start
                        or range_.start > residue_range.end
                    ):
                        matching_elements.append(element)
                        break
        return matching_elements

    def add_element(
        self,
        element: SceneElement,
        parent: Optional[SceneGroup] = None,
        chain_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """Add any scene element (including groups) to the scene."""
        assert element not in self._elements, "Element must not be in the scene"
        assert parent is None or parent in self._elements, "Parent must be in the scene"

        range_set = None
        if chain_id is not None and start is not None and end is not None:
            range_set = ResidueRangeSet([ResidueRange(chain_id, start, end)])

        self._register_element(element, range_set)
        self._set_parent(element, parent)

    def move_element_to_parent(
        self,
        element: SceneElement,
        new_parent: SceneGroup,
    ) -> None:
        """Move an element from its current parent to a new parent group.

        This method preserves any existing residue mappings and ensures proper
        registration in the scene hierarchy.

        Args:
            element: The element to move
            new_parent: The group that will become the element's new parent
        """
        # Store mapping info if it exists
        range_set = self._residue_mappings.get(element)

        assert new_parent in self._elements, "New parent must be in the scene"
        assert element in self._elements, "Element must be in the scene"
        assert isinstance(new_parent, SceneGroup), "New parent must be a SceneGroup"

        # Remove from old parent
        if element._parent:
            element._parent.remove_element(element)
            self._unregister_element(element)

        # Add to new parent
        new_parent.add_element(element)

        # Restore mapping if it existed
        if range_set:
            self._register_element(element, range_set)

    @property
    def root(self) -> SceneGroup:
        """Get the root group of the scene."""
        return self._root_group

    def __iter__(self):
        """Iterate over all structural elements in the scene."""
        return iter(self._elements)

    def move_elements_to_group(
        self,
        elements: list[SceneElement],
        group: SceneGroup,
        parent: Optional[SceneGroup] = None,
    ) -> SceneGroup:
        if group not in self._elements:
            self._register_element(group)

        # Move elements to new group
        for element in elements:
            self.move_element_to_parent(element, group)

        if parent is None:
            parent = self._root_group
        parent.add_element(element=group)

    def __repr__(self) -> str:
        repr = f"Scene(elements={len(self._elements)})"
        for element in self._elements:
            if isinstance(element, SceneGroup) or element.parent is self._root_group:
                repr += f"\n\t{element}"
        return repr
