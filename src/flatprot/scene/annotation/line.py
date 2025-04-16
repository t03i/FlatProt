# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple

from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core import ResidueCoordinate

# Import base classes from the same directory
from .base_annotation import (
    BaseAnnotationElement,
    BaseAnnotationStyle,
)
from ..base_element import SceneGroupType  # Need SceneGroupType for parent hint


# --- Line Annotation Specific Style ---
class LineAnnotationStyle(BaseAnnotationStyle):
    """Style properties specific to LineAnnotation elements."""

    stroke_width: float = Field(
        default=1.0, ge=0, description="Width of the annotation line."
    )
    line_style: Tuple[float, ...] = Field(
        default=(5, 5),
        description="Dash pattern for the line (e.g., (5, 5) for dashed). Empty tuple means solid.",
    )
    connector_color: Color = Field(
        default=Color("#000000"),
        description="Color of the connector circles at the start and end of the line.",
    )
    line_color: Color = Field(
        default=Color("#000000"),
        description="Color of the line.",
    )
    arrowhead_start: bool = Field(
        default=False,
        description="Whether to draw an arrowhead at the start of the line.",
    )
    arrowhead_end: bool = Field(
        default=False,
        description="Whether to draw an arrowhead at the end of the line.",
    )
    connector_radius: float = Field(
        default=2.0,
        ge=0,
        description="Radius of the connector circles at the start and end of the line.",
    )


# --- Line Annotation Scene Element ---
class LineAnnotation(BaseAnnotationElement[LineAnnotationStyle]):
    """Represents an annotation connecting two specific residue coordinates with a line."""

    def __init__(
        self,
        id: str,
        target_coordinates: List[ResidueCoordinate],  # Expects exactly two coordinates
        style: Optional[LineAnnotationStyle] = None,
        label: Optional[str] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a LineAnnotation.

        Args:
            id: A unique identifier for this annotation element.
            target_coordinates: A list containing exactly two ResidueCoordinates
                                defining the start and end points of the line.
            style: The specific style instance for this line annotation.
            label: The label for the annotation.
            parent: The parent SceneGroup in the scene graph, if any.

        Raises:
            ValueError: If `target_coordinates` does not contain exactly two elements.
            TypeError: If elements in `target_coordinates` are not ResidueCoordinate instances.
        """
        if not isinstance(target_coordinates, list) or len(target_coordinates) != 2:
            raise ValueError(
                "LineAnnotation target_coordinates must be a list containing exactly two ResidueCoordinate instances."
            )
        if not all(
            isinstance(coord, ResidueCoordinate) for coord in target_coordinates
        ):
            raise TypeError(
                "All items in target_coordinates must be ResidueCoordinate instances."
            )

        # Call superclass init
        super().__init__(
            id=id,
            target_coordinates=target_coordinates,
            residue_range_set=None,  # Explicitly None, targeting is via coordinates
            style=style,
            label=label,
            parent=parent,
        )

    @property
    def start_coordinate(self) -> ResidueCoordinate:
        """Get the start target coordinate for the line."""
        # target_coordinates is guaranteed to be a list with two elements by __init__
        return self._target_coordinates[0]

    @property
    def end_coordinate(self) -> ResidueCoordinate:
        """Get the end target coordinate for the line."""
        return self._target_coordinates[1]

    @property
    def default_style(self) -> LineAnnotationStyle:
        """Provides the default style for LineAnnotation elements."""
        return LineAnnotationStyle()
