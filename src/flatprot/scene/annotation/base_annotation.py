# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Tuple, Union

import numpy as np
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core.coordinates import ResidueRangeSet, ResidueCoordinate, ResidueRange
from flatprot.core.structure import Structure

# Assuming base_element.py is the correct name now
from ..base_element import BaseSceneElement, BaseSceneStyle, SceneGroupType

# Import CoordinateResolver for type hint
from ..resolver import CoordinateResolver

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

    Stores the original target specification (coordinates, range, or range set)
    and requires the corresponding ResidueRangeSet for the base scene element.
    Requires a concrete style type inheriting from BaseAnnotationStyle.
    """

    def __init__(
        self,
        id: str,  # ID is required for annotations
        target: Union[
            ResidueCoordinate, List[ResidueCoordinate], ResidueRange, ResidueRangeSet
        ],
        label: Optional[str] = None,
        style: Optional[AnnotationStyleType] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a BaseAnnotationElement.

        Subclasses are responsible for constructing the appropriate `residue_range_set`
        based on their specific `target` type before calling this initializer.

        Args:
            id: A unique identifier for this annotation element.
            target: The original target specification (list of coordinates, range, or set).
                    Stored for use by subclasses in `get_coordinates`.
            residue_range_set: The ResidueRangeSet derived from the target, required by
                               the BaseSceneElement for its internal logic (e.g., bounding box).
            label: The label for the annotation.
            style: An optional specific style instance for this annotation.
            parent: The parent SceneGroup in the scene graph, if any.

        Raises:
            TypeError: If the target type is not one of the allowed types.
            ValueError: If residue_range_set is empty.
        """
        # Validate the target type
        if not isinstance(
            target, (ResidueCoordinate, list, ResidueRange, ResidueRangeSet)
        ) or (
            isinstance(target, list)
            and not all(isinstance(item, ResidueCoordinate) for item in target)
        ):
            raise ValueError(
                f"Unsupported target type for annotation: {type(target)}. "
                f"Expected List[ResidueCoordinate], ResidueRange, or ResidueRangeSet."
            )

        self.label = label
        self._target = target  # Store the original target

        # Pass the explicitly provided residue_range_set to the BaseSceneElement constructor
        super().__init__(
            id=id,
            style=style,
            parent=parent,
        )

    @property
    def target(self) -> Union[List[ResidueCoordinate], ResidueRange, ResidueRangeSet]:
        """Get the target specification provided during initialization."""
        return self._target

    @property
    def targets_specific_coordinates(self) -> bool:
        """Check if this annotation targets a list of specific coordinates."""
        return isinstance(self._target, list)

    @abstractmethod
    def get_coordinates(self, resolver: CoordinateResolver) -> np.ndarray:
        """Calculate the renderable coordinates for this annotation.

        Uses the provided CoordinateResolver to find the correct coordinates for
        its target (coordinates, range, or range set) in the context of the scene elements.
        The interpretation of the target depends on the concrete annotation type.

        Args:
            resolver: The CoordinateResolver instance for the scene.

        Returns:
            A NumPy array of coordinates (shape [N, 3], X, Y, Z) suitable for rendering.

        Raises:
            CoordinateCalculationError: If coordinates cannot be resolved.
            TargetResidueNotFoundError: If a target residue is not found.
            # Other specific exceptions possible depending on implementation
        """
        raise NotImplementedError

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
