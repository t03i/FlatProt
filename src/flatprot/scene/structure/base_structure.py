# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

import numpy as np
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core import (
    ResidueRangeSet,
    ResidueCoordinate,
    Structure,
    CoordinateCalculationError,
    logger,
)

# Note: Adjusted import path assuming base.py is in the same directory or parent
from ..base_element import BaseSceneElement, BaseSceneStyle, SceneGroupType

# Generic Type Variable for the specific Structure Style (e.g., HelixStyle)
StructureStyleType = TypeVar("StructureStyleType", bound="BaseStructureStyle")


class BaseStructureStyle(BaseSceneStyle):
    """Base style for elements representing parts of the protein structure."""

    color: Color = Field(
        default=Color((0.5, 0.5, 0.5)),
        description="Default color for the element (hex string). Grey.",
    )
    stroke_color: Color = Field(
        default=Color((0.0, 0.0, 0.0)),
        description="Color for the stroke (hex string). Black.",
    )
    stroke_width: float = Field(
        default=1.0, ge=0.0, description="Line width for stroke."
    )
    opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Opacity for the element."
    )


class BaseStructureSceneElement(
    BaseSceneElement[StructureStyleType], ABC, Generic[StructureStyleType]
):
    """Abstract base class for scene elements representing structural components.

    Automatically generates an ID based on the subclass type and residue range.
    Requires a concrete style type inheriting from BaseStructureStyle.
    """

    def __init__(
        self,
        residue_range_set: ResidueRangeSet,
        style: Optional[StructureStyleType] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes a BaseStructureSceneElement.

        The ID is generated automatically based on the concrete subclass name
        and the provided residue range set.

        Args:
            residue_range_set: The set of residue ranges this element represents.
            style: An optional specific style instance for this element. If None,
                   the default style defined by the subclass's `default_style`
                   property will be used at access time.
            metadata: Optional dictionary for storing arbitrary metadata.
            parent: The parent SceneGroup in the scene graph, if any.
        """
        # Generate the ID *before* calling super().__init__
        # Uses the concrete class's name (e.g., "Helix")
        generated_id = self._generate_id(self.__class__, residue_range_set)

        self.residue_range_set = residue_range_set
        # Pass the potentially None style to the BaseSceneElement constructor.
        # BaseSceneElement's `style` property handles returning the default
        # style if the instance's `_style` attribute is None.
        super().__init__(
            id=generated_id,
            style=style,  # Pass the direct input style (can be None)
            parent=parent,
        )

    @staticmethod
    def _generate_id(cls: type, residue_range_set: ResidueRangeSet) -> str:
        """Generates a unique ID based on class name and residue range set."""
        # Ensure the string representation of the range set is canonical and ID-friendly
        # Replace spaces, commas, and colons to create a valid identifier part.
        # Sorting ranges within the set ensures canonical representation if order matters.
        # Assuming ResidueRangeSet.__str__ provides a consistent, sorted representation.
        range_repr = (
            str(residue_range_set).replace(" ", "").replace(",", "_").replace(":", "-")
        )
        return f"{cls.__name__}-{range_repr}"

    @abstractmethod
    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Retrieve the final 2D + Depth coordinates for rendering this element.

        Implementations should use the element's `residue_range_set` to query
        the provided `structure` object (which is assumed to already contain
        projected 2D + Depth coordinates) and return the relevant slice or
        a simplified representation (e.g., lines for coils) based on these
        pre-projected coordinates.

        Args:
            structure: The core Structure object containing pre-projected
                       2D + Depth coordinate data.

        Returns:
            A NumPy array of 2D + Depth coordinates (shape [N, 3] or similar)
            suitable for rendering (X, Y, Depth).
        """
        raise NotImplementedError

    # Concrete subclasses (Helix, Sheet, etc.) MUST implement default_style
    @property
    @abstractmethod
    def default_style(self) -> StructureStyleType:
        """Provides the default style instance for this specific element type.

        Concrete subclasses (e.g., Helix, Sheet) must implement this property.

        Returns:
            An instance of the specific StyleType (e.g., HelixStyle) for this element.
        """
        raise NotImplementedError

    def get_coordinate_at_residue(
        self, residue: ResidueCoordinate, structure: Structure
    ) -> Optional[np.ndarray]:
        """Retrieves the specific 2D canvas coordinate + Depth corresponding
        to a given residue, derived from the pre-projected coordinates in the
        structure object.

        This default implementation assumes a direct mapping between the residue
        index and the corresponding entry in the main structure.coordinates array.
        Subclasses that implement complex coordinate calculations or simplifications
        in their `get_coordinates` method (e.g., smoothing, interpolation)
        MAY NEED TO OVERRIDE this method to provide the correct mapping to their
        specific rendered representation.

        Args:
            residue: The residue coordinate (chain and index) to find the 2D point for.
            structure: The core Structure object containing pre-projected 2D + Depth data.

        Returns:
            A NumPy array representing the calculated 2D coordinate + Depth (e.g., [X, Y, Depth]),
            or None if the residue is not found or its coordinate cannot be determined.
        """
        try:
            # Check if the residue belongs to the range represented by this element
            if residue not in self.residue_range_set:
                logger.debug(
                    f"Residue {residue} not in range set {self.residue_range_set} for element '{self.id}'"
                )
                return None

            chain = structure[residue.chain_id]

            # Check if the residue index exists in this chain's mapping
            if residue.residue_index not in chain:
                logger.debug(
                    f"Residue {residue} not in chain {chain} for element '{self.id}'"
                )
                return None

            coord_index = chain.coordinate_index(residue.residue_index)
            if not (0 <= coord_index < len(structure.coordinates)):
                struct_id = getattr(structure, "id", "N/A")
                raise CoordinateCalculationError(
                    f"Coordinate index {coord_index} out of bounds for residue {residue} in structure '{struct_id}' (element '{self.id}')."
                )
            return structure.coordinates[coord_index]
        except KeyError:
            logger.debug(
                f"Residue {residue} not in chain {chain} for element '{self.id}'"
            )
            return None
        except (IndexError, AttributeError) as e:
            struct_id = getattr(structure, "id", "N/A")
            raise CoordinateCalculationError(
                f"Error retrieving coordinate for {residue} in structure '{struct_id}' (element '{self.id}'): {e}"
            ) from e

    def get_depth(self, structure: Structure) -> Optional[float]:
        """Calculate the mean depth of this structural element.

        Calculates the mean of the depth values (column 2) of the
        pre-projected coordinates corresponding to the residues in the
        element's residue_range_set.

        Args:
            structure: The core Structure object containing pre-projected
                       2D + Depth coordinate data.

        Returns:
            The mean depth as a float, or None if no coordinates are found.
        """
        # Get coordinates directly from the element's get_coordinates method
        # which handles different element types appropriately
        coords = self.get_coordinates(structure)

        if coords is None or len(coords) == 0:
            return None

        # Extract depth values (Z-coordinate) from the coordinates
        depths = coords[:, 2]

        if len(depths) == 0:
            return None

        return float(np.mean(depths))

    def is_adjacent_to(self, other: "BaseStructureSceneElement") -> bool:
        """Check if this element is adjacent to another element.

        Args:
            other: The other element to check adjacency with.

        Returns:
            True if the elements are adjacent, False otherwise.
        """
        if not isinstance(other, BaseStructureSceneElement):
            raise TypeError(f"Cannot check adjacency with {type(other)}")

        return self.residue_range_set.is_adjacent_to(other.residue_range_set)

    @abstractmethod
    def get_start_connection_point(self, structure: Structure) -> Optional[np.ndarray]:
        """Get the 2D coordinate (X, Y) of the start connection point.

        This is typically the coordinate corresponding to the first residue
        in the element's range, projected onto the 2D canvas.

        Args:
            structure: The core Structure object with pre-projected coordinates.

        Returns:
            A NumPy array [X, Y] or None if not applicable/determinable.
        """
        raise NotImplementedError

    @abstractmethod
    def get_end_connection_point(self, structure: Structure) -> Optional[np.ndarray]:
        """Get the 2D coordinate (X, Y) of the end connection point.

        This is typically the coordinate corresponding to the last residue
        in the element's range, projected onto the 2D canvas.

        Args:
            structure: The core Structure object with pre-projected coordinates.

        Returns:
            A NumPy array [X, Y] or None if not applicable/determinable.
        """
        raise NotImplementedError
