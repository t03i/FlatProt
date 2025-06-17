# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
from pydantic import Field

from flatprot.core import (
    ResidueCoordinate,
)

# Import base classes from the same directory
from .base_annotation import (
    BaseAnnotationElement,
    BaseAnnotationStyle,
)
from ..base_element import SceneGroupType

# Import CoordinateResolver
from ..resolver import CoordinateResolver


# --- Point Annotation Specific Style ---
class PointAnnotationStyle(BaseAnnotationStyle):
    """Style properties specific to PointAnnotation elements."""

    marker_radius: float = Field(
        default=5.0,
        ge=0,
        description="Radius of the point marker.",
    )


# --- Point Annotation Scene Element ---
class PointAnnotation(BaseAnnotationElement[PointAnnotationStyle]):
    """Represents an annotation marking a single residue coordinate."""

    def __init__(
        self,
        id: str,
        target: ResidueCoordinate,  # Expects a single coordinate
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
        if not isinstance(target, ResidueCoordinate):
            raise TypeError(
                "target_coordinate must be a single ResidueCoordinate instance."
            )

        # Call superclass init, passing the single coordinate in a list
        super().__init__(
            id=id,
            target=target,  # Base class expects a list
            style=style,
            label=label,
            parent=parent,
        )

    @property
    def target_coordinate(self) -> ResidueCoordinate:
        """Get the specific target coordinate for this point annotation."""
        # target_coordinates is guaranteed to be a list with one element by __init__
        return self.target

    @property
    def default_style(self) -> PointAnnotationStyle:
        """Provides the default style for PointAnnotation elements."""
        return PointAnnotationStyle()

    def get_coordinates(self, resolver: CoordinateResolver) -> np.ndarray:
        """Calculate the coordinates for the point annotation marker.

        Uses the CoordinateResolver to find the rendered coordinate of the target residue.

        Args:
            resolver: The CoordinateResolver instance for the scene.

        Returns:
            A NumPy array of shape [1, 3] containing the (X, Y, Z) coordinates
            of the target point.

        Raises:
            CoordinateCalculationError: If the coordinate cannot be resolved.
            TargetResidueNotFoundError: If the target residue is not found.
        """
        target_res = self.target_coordinate
        # Delegate resolution to the resolver
        point = resolver.resolve(target_res)
        # Resolver handles errors, so point should be valid if no exception was raised
        return np.array([point])
