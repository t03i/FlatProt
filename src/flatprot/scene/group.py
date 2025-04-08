# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Dict, Any
from flatprot.core import ResidueRangeSet, ResidueRange, Structure
from .base_element import BaseSceneElement, BaseSceneStyle


class GroupStyle(BaseSceneStyle):
    """Style specific to SceneGroup elements."""

    # Example: Add group-specific style attributes if needed later
    # border_color: Optional[str] = Field(default=None, description="Optional border color for the group bounds.")
    pass  # Inherits visibility and opacity, add more as needed


class SceneGroup(BaseSceneElement[GroupStyle]):
    """Represents a group node in the scene graph.

    A SceneGroup can contain other SceneElements (including other SceneGroups)
    and apply transformations to them collectively.
    Its residue range is the union of its children's ranges.
    """

    def __init__(
        self,
        id: str,
        children: Optional[List[BaseSceneElement]] = None,
        transforms: Optional[Dict[str, Any]] = None,
        style: Optional[GroupStyle] = None,
        parent: Optional["SceneGroup"] = None,  # Type hint refers to its own class
    ):
        """Initializes a SceneGroup.

        Args:
            id: A unique identifier for this group.
            children: An optional list of initial child elements.
            transforms: Optional dictionary defining transformations (e.g., translate, scale).
            style: An optional specific style instance for this group.
            metadata: Optional dictionary for storing arbitrary metadata.
            parent: The parent SceneGroup in the scene graph, if any.
        """
        self._children: List[BaseSceneElement] = []
        self.transforms: Dict[str, Any] = transforms or {}

        # Initialize BaseSceneElement with an empty or calculated range set
        # The range set will be updated as children are added/removed.
        initial_range_set = self._calculate_combined_range_set(children or [])
        super().__init__(id, initial_range_set, style, parent)

        # Add initial children after super().__init__ has run
        if children:
            for child in children:
                # Use add_child to ensure parentage and range set are updated
                self.add_child(child)

    @property
    def children(self) -> List[BaseSceneElement]:
        """Get the list of direct child elements."""
        return self._children

    def _calculate_combined_range_set(
        self, elements: List[BaseSceneElement]
    ) -> ResidueRangeSet:
        """Calculates the union of residue ranges from a list of elements."""
        all_ranges: List[ResidueRange] = []
        for element in elements:
            if element.residue_range_set:
                all_ranges.extend(element.residue_range_set.ranges)

        # Creating a new ResidueRangeSet handles sorting and potentially merging/simplifying ranges
        # (depending on ResidueRangeSet's internal logic, assuming it handles this)
        # If ResidueRangeSet doesn't merge, this will just be a collection of all child ranges.
        return ResidueRangeSet(all_ranges)

    def add_child(self, element: BaseSceneElement) -> None:
        """Adds a SceneElement as a child of this group.

        Args:
            element: The SceneElement to add.

        Raises:
            ValueError: If the element is already a child of another group or this group.
            TypeError: If the element is not a BaseSceneElement instance.
        """
        if not isinstance(element, BaseSceneElement):
            raise TypeError("Child must be an instance of BaseSceneElement.")
        if element._parent is not None:
            raise ValueError(
                f"Element '{element.id}' already has a parent ('{element.parent.id}'). "
                f"Remove it from its current parent before adding."
            )
        if element is self:
            raise ValueError("Cannot add a group as a child of itself.")
        # Prevent circular parenting by walking up the tree
        curr = self
        while curr is not None:
            if curr is element:
                raise ValueError("Cannot create circular parent relationships.")
            curr = curr.parent

        element._set_parent(self)  # Use internal setter
        self._children.append(element)
        # Update the group's residue range set
        self.residue_range_set = self._calculate_combined_range_set(self._children)

    def remove_child(self, element: BaseSceneElement) -> None:
        """Removes a SceneElement from this group's children.

        Args:
            element: The SceneElement to remove.

        Raises:
            ValueError: If the element is not a direct child of this group.
        """
        if element not in self._children:
            raise ValueError(
                f"Element '{element.id}' is not a direct child of group '{self.id}'."
            )

        element._set_parent(None)  # Use internal setter
        self._children.remove(element)
        # Update the group's residue range set
        self.residue_range_set = self._calculate_combined_range_set(self._children)

    @property
    def default_style(self) -> GroupStyle:
        """Provides the default style instance for SceneGroup."""
        return GroupStyle()

    def get_coordinates(self, structure: Structure) -> None:
        """Groups themselves do not have coordinates.

        Returns:
            None.
        """
        return None

    def __iter__(self):
        """Iterate over the children of this group."""
        return iter(self._children)

    def __repr__(self) -> str:
        """Provide a string representation of the scene group."""
        parent_id = f"'{self._parent.id}'" if self._parent else None
        transform_repr = ", ".join(f"{k}={v}" for k, v in self.transforms.items())
        return (
            f"<{self.__class__.__name__} id='{self.id}' children={len(self._children)} "
            f"parent={parent_id} transforms={transform_repr}>"
        )

    def get_depth(self, structure: Structure) -> Optional[float]:
        """Groups themselves do not have an inherent depth.

        Returns:
            None.
        """
        return None
