# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

from flatprot.style import StyleType
from .structure import StructureSceneElement
from .annotations import Annotation, GroupAnnotation
from .elements import SceneGroup, SceneElement


@dataclass
class ResidueMapping:
    """Mapping information for a structure element"""

    chain_id: str
    start: int
    end: int

    @property
    def residue_range(self) -> range:
        return range(self.start, self.end + 1)


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
        self._residue_mappings: dict[StructureSceneElement, ResidueMapping] = {}
        self._residue_ranges: dict[str, list[tuple[range, StructureSceneElement]]] = {}
        self._root_group = SceneGroup(id="root")

    def _register_element(
        self,
        element: SceneElement,
        residue_mapping: ResidueMapping | None = None,
    ) -> None:
        """Internal method to register a structural element in the scene.

        Updates residue mappings if the element is a StructureSceneElement.
        """
        if residue_mapping is not None:
            self._residue_mappings[element] = residue_mapping
        self._elements.append(element)

    def _unregister_element(self, element: SceneElement) -> ResidueMapping | None:
        """Internal method to unregister a structural element from the scene."""
        mapping = None
        if element in self._residue_mappings:
            mapping = self._residue_mappings[element]
            del self._residue_mappings[element]

        self._elements.remove(element)
        return mapping

    def _set_parent(self, element: SceneElement, parent: SceneGroup) -> None:
        """Internal method to set the parent of an element."""
        assert element in self._elements, "Element must be registered in the scene"
        assert parent is None or parent in self._elements, (
            "Parent group must be registered in the scene"
        )
        assert parent is None or isinstance(parent, SceneGroup), (
            "Parent must be a SceneGroup instance"
        )

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
        self, chain_id: str, residue: int
    ) -> list[StructureSceneElement]:
        """Get all structural elements containing the specified residue.

        Args:
            chain_id: The chain ID to search for elements
            residue: The residue number to search for

        Returns:
            A list of StructureSceneElements that contain the specified residue
        """
        matching_elements = []
        for element, mapping in self._residue_mappings.items():
            if (
                isinstance(element, StructureSceneElement)
                and mapping.chain_id == chain_id
                and residue in mapping.residue_range
            ):
                matching_elements.append(element)
        return matching_elements

    def get_elements_for_residue_range(
        self, chain_id: str, start: int, end: int
    ) -> list[StructureSceneElement]:
        """Get all structural elements containing the specified residue range.

        Args:
            chain_id: The chain ID to search for elements
            start: The start residue number to search for
            end: The end residue number to search for

        Returns:
            A list of StructureSceneElements that contain the specified residue range
        """
        # Return empty list if start > end
        if start > end:
            return []

        matching_elements = []
        for element, mapping in self._residue_mappings.items():
            if (
                isinstance(element, StructureSceneElement)
                and mapping.chain_id == chain_id
                and (
                    # Check if ranges overlap
                    (mapping.start <= start <= mapping.end)
                    or (mapping.start <= end <= mapping.end)
                    or (start <= mapping.start and end >= mapping.end)
                )
            ):
                matching_elements.append(element)
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

        residue_mapping = None
        if chain_id is not None and start is not None and end is not None:
            residue_mapping = ResidueMapping(chain_id, start, end)

        self._register_element(element, residue_mapping)
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
        mapping = self._residue_mappings.get(element)

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
        if mapping:
            self._register_element(
                element, mapping.chain_id, mapping.start, mapping.end
            )

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
