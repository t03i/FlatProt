# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Literal

import numpy as np
from pydantic import Field

from flatprot.core import ResidueCoordinate, Structure, TargetResidueNotFoundError

# Import base classes from the same directory
from .base_annotation import (
    BaseAnnotationElement,
    BaseAnnotationStyle,
)
from ..base_element import SceneGroupType


# --- Point Annotation Specific Style ---
class PointAnnotationStyle(BaseAnnotationStyle):
    """Style properties specific to PointAnnotation elements."""

    marker_shape: Literal["circle", "square", "triangle", "diamond"] = Field(
        default="circle", description="Shape of the marker."
    )
    marker_radius: float = Field(
        default=5.0,
        ge=0,
        description="Radius of the point marker.",
    )
    # Inherits color, offset, label etc. from BaseAnnotationStyle


# --- Point Annotation Scene Element ---
class PointAnnotation(BaseAnnotationElement[PointAnnotationStyle]):
    """Represents an annotation marking a single residue coordinate."""

    def __init__(
        self,
        id: str,
        target_coordinate: ResidueCoordinate,  # Expects a single coordinate
        label: Optional[str] = None,
        style: Optional[PointAnnotationStyle] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a PointAnnotation.

        Args:
            id: A unique identifier for this annotation element.
            target_coordinate: The specific residue coordinate this annotation targets.
            style: An optional specific style instance for this annotation.
            metadata: Optional dictionary for storing arbitrary metadata.
            parent: The parent SceneGroup in the scene graph, if any.
        """
        if not isinstance(target_coordinate, ResidueCoordinate):
            raise TypeError(
                "target_coordinate must be a single ResidueCoordinate instance."
            )

        # Call superclass init, passing the single coordinate in a list
        super().__init__(
            id=id,
            target_coordinates=[target_coordinate],  # Base class expects a list
            residue_range_set=None,  # Explicitly None, targeting is via coordinate
            style=style,
            label=label,
            parent=parent,
        )

    @property
    def target_coordinate(self) -> ResidueCoordinate:
        """Get the specific target coordinate for this point annotation."""
        # target_coordinates is guaranteed to be a list with one element by __init__
        return self._target_coordinates[0]

    @property
    def default_style(self) -> PointAnnotationStyle:
        """Provides the default style for PointAnnotation elements."""
        return PointAnnotationStyle()

    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Calculate the coordinates for the point annotation marker.

        Retrieves the 3D coordinates of the target residue from the provided structure.

        Args:
            structure: The core Structure object containing coordinate data.

        Returns:
            A NumPy array of shape [1, 3] containing the (X, Y, Z) coordinates
            of the target point, or None if the coordinate cannot be resolved.
        """
        target_res = self.target_coordinate
        point = structure.get_coordinate_at_residue(target_res)

        if point is None:
            raise TargetResidueNotFoundError(structure, target_res)

        return np.array([point])
