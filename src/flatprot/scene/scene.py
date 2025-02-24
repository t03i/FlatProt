# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional


from flatprot.style import StyleType
from .structure import StructureSceneElement
from .annotations import Annotation, GroupAnnotation
from .elements import SceneGroup, SceneElement


class Scene:
    """Root container for all scene elements and groups.

    The Scene maintains both:
    1. A flat list of structural elements with residue mapping for annotations
    2. A hierarchical tree of groups for SVG rendering
    """

    def __init__(self):
        # For annotation/mapping purposes
        self._elements: list[StructureSceneElement] = []
        self._residue_ranges: dict[str, list[tuple[range, StructureSceneElement]]] = {}

        # For rendering hierarchy
        self._root_group: SceneGroup = SceneGroup(id="root")

    def add_structural_element(
        self, element: StructureSceneElement, parent: Optional[SceneGroup] = None
    ) -> None:
        """Add a structural element and update residue mapping.

        Args:
            element: The structural element to add
            parent: The group to add the element to (defaults to root)
        """
        # Update residue mapping
        chain_id = element._chain_id
        start = element._start
        end = element._end

        if chain_id not in self._residue_ranges:
            self._residue_ranges[chain_id] = []
        self._residue_ranges[chain_id].append((range(start, end + 1), element))
        self._elements.append(element)

        # Add to rendering hierarchy
        if parent is None:
            parent = self._root_group
        parent.add_element(element)

    def get_elements_for_residue(
        self, chain_id: str, residue: int
    ) -> list[StructureSceneElement]:
        """Get all structural elements containing the specified residue."""
        if chain_id not in self._residue_ranges:
            return []
        return [elem for rng, elem in self._residue_ranges[chain_id] if residue in rng]

    def create_group(
        self, id: str, parameters: dict = {}, parent: Optional[SceneGroup] = None
    ) -> SceneGroup:
        """Create a new empty group in the rendering hierarchy.

        Args:
            id: Identifier for the new group
            parameters: Parameters for the new group
            parent: Parent group (defaults to root)
        """
        group = SceneGroup(id=id, **parameters)
        if parent is None:
            parent = self._root_group
        parent.add_element(group)
        return group

    def create_subgroup(
        self, parent_group: SceneGroup, elements: list[SceneElement], id: str
    ) -> SceneGroup:
        """Create a new group containing the specified elements.

        This will move the elements from their current parent to be children
        of the new group. The new group will be a child of parent_group.

        Args:
            parent_group: The group that will contain the new subgroup
            elements: Elements to move into the new group
            id: Identifier for the new group
        """
        new_group = SceneGroup(id=id)

        # Move elements to new group
        for element in elements:
            if element._parent:
                element._parent.remove_element(element)
            new_group.add_element(element)

        # Add new group to parent
        parent_group.add_element(new_group)
        return new_group

    def get_parent_group(self, element: SceneElement) -> Optional[SceneGroup]:
        """Get the parent group of an element."""
        return element._parent

    @property
    def root(self) -> SceneGroup:
        """Get the root group of the scene."""
        return self._root_group

    def __iter__(self):
        """Iterate over all structural elements in the scene."""
        return iter(self._elements)

    def add_annotation(
        self,
        annotation_type: str,
        content: Any,
        targets: list[SceneElement] | SceneElement,
        parent: Optional[SceneGroup] = None,
        style_type: Optional[StyleType] = None,
    ) -> Annotation:
        """Add a regular annotation that references its targets."""
        if not isinstance(targets, list):
            targets = [targets]

        annotation = Annotation(
            annotation_type=annotation_type,
            content=content,
            targets=targets,
            style_manager=self._root_group.style,
            style_type=style_type,
        )

        if parent is None:
            parent = self._root_group
        parent.add_element(annotation)
        return annotation

    def add_group_annotation(
        self,
        annotation_type: str,
        content: Any,
        elements: list[SceneElement],
        parent: Optional[SceneGroup] = None,
        style_type: Optional[StyleType] = None,
    ) -> GroupAnnotation:
        """Add a group annotation that contains its elements."""
        group = GroupAnnotation(
            annotation_type=annotation_type,
            content=content,
            elements=elements,
            style_manager=self._root_group.style,
            style_type=style_type,
        )

        # Move elements to new group
        for element in elements:
            old_parent = self.get_parent_group(element)
            if old_parent:
                old_parent.remove_element(element)

        if parent is None:
            parent = self._root_group
        parent.add_element(group)
        return group
