# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Tuple

import numpy as np
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core.coordinates import ResidueRangeSet, ResidueCoordinate, ResidueRange
from flatprot.core.structure import Structure

# Assuming base_element.py is the correct name now
from ..base_element import BaseSceneElement, BaseSceneStyle, SceneGroupType

# Generic Type Variable for the specific Annotation Style
AnnotationStyleType = TypeVar("AnnotationStyleType", bound="BaseAnnotationStyle")


class BaseAnnotationStyle(BaseSceneStyle):
    """Base style for annotation elements."""

    color: Color = Field(
        default=Color((1.0, 0.0, 0.0)),
        description="Default color for the annotation (hex string). Red.",
    )
    offset: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="2D offset (x, y) from the anchor point in canvas units.",
    )
    label_offset: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="2D offset (x, y) from the label anchor point in canvas units.",
    )
    label_color: Color = Field(
        default=Color((0.0, 0.0, 0.0)),
        description="Default color for the label (hex string). Black.",
    )
    label_font_size: float = Field(
        default=12.0,
        description="Font size for the label.",
    )
    label_font_weight: str = Field(
        default="normal",
        description="Font weight for the label.",
    )
    label_font_family: str = Field(
        default="Arial",
        description="Font family for the label.",
    )
    label: Optional[str] = Field(
        default=None, description="Optional text label for the annotation."
    )


class BaseAnnotationElement(
    BaseSceneElement[AnnotationStyleType], ABC, Generic[AnnotationStyleType]
):
    """Abstract base class for scene elements representing annotations.

    Annotations can target specific residues/points or broader residue ranges.
    Requires a concrete style type inheriting from BaseAnnotationStyle.
    """

    def __init__(
        self,
        id: str,  # ID is required for annotations
        residue_range_set: Optional[ResidueRangeSet] = None,
        target_coordinates: Optional[List[ResidueCoordinate]] = None,
        label: Optional[str] = None,
        style: Optional[AnnotationStyleType] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a BaseAnnotationElement.

        Exactly one of `residue_range_set` or `target_coordinates` must be provided
        to define what the annotation refers to.

        Args:
            id: A unique identifier for this annotation element.
            residue_range_set: The set of residue ranges this annotation targets (for area-like annotations).
            target_coordinates: A list of specific residue coordinates this annotation targets (for point/line annotations).
            label: The label for the annotation.
            style: An optional specific style instance for this annotation.
            parent: The parent SceneGroup in the scene graph, if any.

        Raises:
            ValueError: If neither or both targeting arguments are provided.
        """

        self.label = label
        if (residue_range_set is None and target_coordinates is None) or (
            residue_range_set is not None and target_coordinates is not None
        ):
            raise ValueError(
                "Exactly one of 'residue_range_set' or 'target_coordinates' must be provided."
            )

        self._target_coordinates: Optional[List[ResidueCoordinate]] = target_coordinates

        # Determine the effective residue_range_set for the superclass init
        effective_range_set: ResidueRangeSet
        if residue_range_set is not None:
            effective_range_set = residue_range_set
        else:  # Derive from target_coordinates
            # Create a minimal ResidueRangeSet covering the target points
            ranges = [
                ResidueRange(coord.chain_id, coord.residue_index, coord.residue_index)
                for coord in target_coordinates
            ]
            effective_range_set = ResidueRangeSet(ranges)

        # Pass the potentially None style to the BaseSceneElement constructor
        super().__init__(
            id=id,
            residue_range_set=effective_range_set,
            style=style,
            parent=parent,
        )

    @property
    def target_coordinates(self) -> Optional[List[ResidueCoordinate]]:
        """Get the specific target coordinates, if defined."""
        return self._target_coordinates

    @property
    def targets_specific_coordinates(self) -> bool:
        """Check if this annotation targets specific coordinates rather than a range set."""
        return self._target_coordinates is not None

    @abstractmethod
    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Return the anchor coordinates for this annotation.

        Annotations derive their position from other elements or specific
        coordinates. This method, in the base class, does not provide
        renderable coordinates itself. The renderer should use target information
        to determine anchor points.

        Args:
            structure: The core Structure object (unused in this base implementation).

        Returns:
            None, as annotations don't have intrinsic renderable coordinates.
        """
        return None

    # Concrete subclasses (Marker, Line, Area) MUST implement default_style
    @property
    @abstractmethod
    def default_style(self) -> AnnotationStyleType:
        """Provides the default style instance for this specific annotation type.

        Concrete subclasses must implement this property.

        Returns:
            An instance of the specific AnnotationStyleType for this element.
        """
        raise NotImplementedError

    def get_depth(self, structure: Structure) -> Optional[float]:
        """Return a fixed high depth value for annotations.

        This ensures annotations are rendered on top of other elements
        when sorted by depth (ascending).

        Args:
            structure: The core Structure object (unused).

        Returns:
            A very large float value (infinity).
        """
        # Return positive infinity to ensure annotations are sorted last (drawn on top)
        # when using ascending sort order for depth. Adjust if sort order is descending.
        return float("inf")
