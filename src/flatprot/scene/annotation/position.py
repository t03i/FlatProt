# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union
from enum import Enum

import numpy as np
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core import (
    ResidueCoordinate,
    ResidueRange,
    ResidueRangeSet,
)

# Import base classes from the same directory
from .base_annotation import (
    BaseAnnotationElement,
    BaseAnnotationStyle,
)
from ..base_element import SceneGroupType

# Import CoordinateResolver
from ..resolver import CoordinateResolver


class PositionType(str, Enum):
    """Types of position annotations."""

    N_TERMINUS = "n_terminus"
    C_TERMINUS = "c_terminus"
    RESIDUE_NUMBER = "residue_number"


# --- Position Annotation Specific Style ---
class PositionAnnotationStyle(BaseAnnotationStyle):
    """Style properties specific to PositionAnnotation elements."""

    font_size: float = Field(
        default=8.0,
        ge=1.0,
        description="Font size for position text.",
    )
    font_weight: str = Field(
        default="normal",
        description="Font weight for position text (normal, bold).",
    )
    font_family: str = Field(
        default="Arial, sans-serif",
        description="Font family for position text.",
    )
    text_offset: float = Field(
        default=5.0,
        ge=0.0,
        description="Offset distance from structure element in pixels.",
    )
    show_terminus: bool = Field(
        default=True,
        description="Whether to show N/C terminus labels.",
    )
    show_residue_numbers: bool = Field(
        default=True,
        description="Whether to show residue numbers on secondary structures.",
    )
    terminus_font_size: float = Field(
        default=10.0,
        ge=1.0,
        description="Font size for N/C terminus labels.",
    )
    terminus_font_weight: str = Field(
        default="bold",
        description="Font weight for N/C terminus labels.",
    )
    font_color: Color = Field(
        default=Color((0.5, 0.5, 0.5)),  # Gray for residue numbers
        description="Font color for residue numbers.",
    )
    terminus_font_color: Color = Field(
        default=Color((0.0, 0.0, 0.0)),  # Black for N/C terminus
        description="Font color for N/C terminus labels.",
    )


# --- Position Annotation Scene Element ---
class PositionAnnotation(BaseAnnotationElement[PositionAnnotationStyle]):
    """Represents a position annotation showing residue numbers or terminus labels."""

    def __init__(
        self,
        id: str,
        target: Union[ResidueCoordinate, ResidueRange, ResidueRangeSet],
        position_type: PositionType,
        text: str,
        style: Optional[PositionAnnotationStyle] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a PositionAnnotation.

        Args:
            id: A unique identifier for this annotation element.
            target: The target residue(s) for this position annotation.
            position_type: The type of position annotation (terminus or residue number).
            text: The text to display (e.g., "N", "C", "42", "156").
            style: An optional specific style instance for this annotation.
            parent: The parent SceneGroup in the scene graph, if any.
        """
        self.position_type = position_type
        self.text = text

        # Call superclass init
        super().__init__(
            id=id,
            target=target,
            label=text,  # Use text as label
            style=style,
            parent=parent,
        )

    @property
    def default_style(self) -> PositionAnnotationStyle:
        """Provides the default style for PositionAnnotation elements."""
        return PositionAnnotationStyle()

    def get_coordinates(self, resolver: CoordinateResolver) -> np.ndarray:
        """Calculate the coordinates for the position annotation text.

        Uses the CoordinateResolver to find the rendered coordinate where the text should be placed.

        Args:
            resolver: The CoordinateResolver instance for the scene.

        Returns:
            A NumPy array of shape [1, 3] containing the (X, Y, Z) coordinates
            where the text should be placed.

        Raises:
            CoordinateCalculationError: If the coordinate cannot be resolved.
            TargetResidueNotFoundError: If the target residue is not found.
        """
        if isinstance(self.target, ResidueCoordinate):
            # Single residue coordinate
            point = resolver.resolve(self.target)
            return np.array([point])
        elif isinstance(self.target, ResidueRange):
            # For a range, use the appropriate end based on position type
            if self.position_type == PositionType.N_TERMINUS:
                # Use start of range
                start_coord = ResidueCoordinate(
                    self.target.chain_id,
                    self.target.start,
                    None,  # residue_type not needed for coordinate resolution
                    0,  # coordinate_index will be resolved
                )
                point = resolver.resolve(start_coord)
            else:
                # Use end of range (C_TERMINUS or RESIDUE_NUMBER)
                end_coord = ResidueCoordinate(
                    self.target.chain_id, self.target.end, None, 0
                )
                point = resolver.resolve(end_coord)
            return np.array([point])
        elif isinstance(self.target, ResidueRangeSet):
            # For a range set, use the first range
            if not self.target.ranges:
                raise ValueError(
                    f"Empty ResidueRangeSet for position annotation {self.id}"
                )
            first_range = self.target.ranges[0]
            if self.position_type == PositionType.N_TERMINUS:
                start_coord = ResidueCoordinate(
                    first_range.chain_id, first_range.start, None, 0
                )
                point = resolver.resolve(start_coord)
            else:
                # Use end of last range for C_TERMINUS
                last_range = self.target.ranges[-1]
                end_coord = ResidueCoordinate(
                    last_range.chain_id, last_range.end, None, 0
                )
                point = resolver.resolve(end_coord)
            return np.array([point])
        else:
            raise ValueError(
                f"Unsupported target type for position annotation: {type(self.target)}"
            )

    def get_display_properties(self) -> dict:
        """Get properties for text display based on position type."""
        if self.position_type in [PositionType.N_TERMINUS, PositionType.C_TERMINUS]:
            return {
                "font_size": self.style.terminus_font_size,
                "font_weight": self.style.terminus_font_weight,
                "font_family": self.style.font_family,
                "font_color": self.style.terminus_font_color,
                "text": self.text,
                "offset": self.style.text_offset,
            }
        else:  # RESIDUE_NUMBER
            return {
                "font_size": self.style.font_size,
                "font_weight": self.style.font_weight,
                "font_family": self.style.font_family,
                "font_color": self.style.font_color,
                "text": self.text,
                "offset": self.style.text_offset,
            }
