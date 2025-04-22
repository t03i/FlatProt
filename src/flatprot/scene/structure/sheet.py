# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
from pydantic import Field
from pydantic_extra_types.color import Color

from flatprot.core import (
    ResidueRangeSet,
    ResidueCoordinate,
    Structure,
    CoordinateCalculationError,
)
from ..base_element import SceneGroupType
from .base_structure import (
    BaseStructureSceneElement,
    BaseStructureStyle,
)


# --- Sheet Specific Style ---
class SheetStyle(BaseStructureStyle):
    """Style properties specific to Sheet elements.

    Defines properties for rendering beta sheets as triangular arrows.
    """

    # Override inherited defaults
    color: Color = Field(
        default=Color("#0000ff"), description="Default color for sheet (blue)."
    )
    stroke_width: float = Field(
        default=1.0, description="Base width of the sheet arrow."
    )
    simplified_width: float = Field(
        default=2,
        description="Width to use for simplified sheet rendering (line only).",
    )
    # Sheet-specific attributes
    arrow_width: float = Field(
        default=8.0,
        description="Factor to multiply linewidth by for the arrowhead base width.",
    )
    min_sheet_length: int = Field(
        default=3,
        ge=1,
        description="Minimum number of residues required to draw an arrow shape instead of a line.",
    )


# --- Sheet Scene Element ---
class SheetSceneElement(BaseStructureSceneElement[SheetStyle]):
    """Represents a Beta Sheet segment, visualized as a triangular arrow."""

    def __init__(
        self,
        residue_range_set: ResidueRangeSet,
        style: Optional[SheetStyle] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes the SheetSceneElement."""
        super().__init__(residue_range_set, style, parent)
        # Cache for the calculated arrow coordinates and original length
        self._cached_display_coords: Optional[np.ndarray] = None
        self._original_coords_len: Optional[int] = None

    @property
    def default_style(self) -> SheetStyle:
        """Provides the default style for Sheet elements."""
        return SheetStyle()

    def _get_original_coords_slice(self, structure: Structure) -> Optional[np.ndarray]:
        """Helper to extract the original coordinate slice for this sheet."""
        coords_list = []
        if not self.residue_range_set.ranges:
            raise CoordinateCalculationError(
                f"Cannot get coordinates for Sheet '{self.id}': no residue ranges defined."
            )
        sheet_range = self.residue_range_set.ranges[0]

        try:
            chain = structure[sheet_range.chain_id]
            for res_idx in range(sheet_range.start, sheet_range.end + 1):
                if res_idx in chain:
                    coord_idx = chain.coordinate_index(res_idx)
                    if 0 <= coord_idx < len(structure.coordinates):
                        coords_list.append(structure.coordinates[coord_idx])
                    else:
                        raise CoordinateCalculationError(
                            f"Sheet '{self.id}': Coordinate index {coord_idx} out of bounds for residue {sheet_range.chain_id}:{res_idx}."
                        )
                else:
                    raise CoordinateCalculationError(
                        f"Sheet '{self.id}': Residue {sheet_range.chain_id}:{res_idx} not found in chain coordinate map."
                    )
        except (KeyError, IndexError, AttributeError) as e:
            raise CoordinateCalculationError(
                f"Error fetching coordinates for sheet '{self.id}': {e}"
            ) from e

        return np.array(coords_list) if coords_list else None

    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Retrieve the 2D + Depth coordinates for the sheet arrow.

        Calculates the three points (arrow base left, base right, tip) based
        on the start and end points of the pre-projected coordinate slice.
        Handles minimum length requirement.

        Args:
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array of the arrow coordinates (shape [3, 3] or [2, 3])
            containing [X, Y, Depth] for each point.
        """
        if self._cached_display_coords is not None:
            return self._cached_display_coords

        original_coords = self._get_original_coords_slice(structure)
        if original_coords is None or len(original_coords) == 0:
            self._cached_display_coords = None
            self._original_coords_len = 0
            return None

        self._original_coords_len = len(original_coords)

        # If only one point, cannot draw line or arrow
        if self._original_coords_len == 1:
            self._cached_display_coords = np.array(
                [original_coords[0]]
            )  # Return the single point
            return self._cached_display_coords

        # Use only X, Y for shape calculation, keep Z (depth)
        start_point_xy = original_coords[0, :2]
        end_point_xy = original_coords[-1, :2]

        # Use average depth of start/end for the base, end depth for tip
        start_depth = original_coords[0, 2]
        end_depth = original_coords[-1, 2]
        avg_base_depth = (start_depth + end_depth) / 2.0

        direction = end_point_xy - start_point_xy
        length = np.linalg.norm(direction)

        # If too short or degenerate, return a simple line (start and end points)
        if length < 1e-6 or self._original_coords_len < self.style.min_sheet_length:
            # Return original start and end points (X, Y, Depth)
            self._cached_display_coords = np.array(
                [original_coords[0], original_coords[-1]]
            )
            return self._cached_display_coords

        # Normalize direction vector (only need X, Y)
        direction /= length

        # Calculate perpendicular vector in 2D
        perp = np.array([-direction[1], direction[0]])
        arrow_base_half_width = self.style.arrow_width / 2.0

        # Calculate arrow base points (X, Y)
        left_point_xy = start_point_xy + perp * arrow_base_half_width
        right_point_xy = start_point_xy - perp * arrow_base_half_width

        # Combine XY with Depth
        left_point = np.append(left_point_xy, avg_base_depth)
        right_point = np.append(right_point_xy, avg_base_depth)
        tip_point = np.append(end_point_xy, end_depth)  # Tip uses depth of last residue

        self._cached_display_coords = np.array([left_point, right_point, tip_point])
        return self._cached_display_coords

    def get_coordinate_at_residue(
        self, residue: ResidueCoordinate, structure: Structure
    ) -> Optional[np.ndarray]:
        """Retrieves the specific 2D coordinate + Depth corresponding to a residue
        along the central axis of the sheet arrow representation.

        Interpolates along the axis from the base midpoint to the tip, or along
        the line if the arrow shape is not drawn.

        Args:
            residue: The residue coordinate (chain and index) to find the point for.
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array [X, Y, Depth] interpolated along the sheet axis.
        """
        # 1. Ensure display coordinates are calculated and length is known
        display_coords = self.get_coordinates(structure)
        if (
            display_coords is None
            or self._original_coords_len is None
            or self._original_coords_len == 0
        ):
            return None

        # 2. Check if residue is within the element's range
        if residue not in self.residue_range_set:
            return None
        # Assuming single continuous range for sheet element representation
        element_range = self.residue_range_set.ranges[0]
        if residue.chain_id != element_range.chain_id:
            return None

        # 3. Calculate the 0-based index within the original sequence length
        try:
            original_sequence_index = residue.residue_index - element_range.start
            # Validate index against the original length before simplification/arrow calc
            if not (0 <= original_sequence_index < self._original_coords_len):
                raise CoordinateCalculationError(
                    f"Residue index {original_sequence_index} derived from {residue} is out of original bounds [0, {self._original_coords_len}) for element {self.id}."
                )
        except Exception as e:
            raise CoordinateCalculationError(
                f"Error calculating original sequence index for {residue} in element {self.id}: {e}"
            ) from e

        # Handle single point case
        if self._original_coords_len == 1:
            return display_coords[
                0
            ]  # Return the single point calculated by get_coordinates

        # 4. Handle the case where a line was drawn (display_coords has 2 points)
        if len(display_coords) == 2:
            # Simple linear interpolation between the start and end points of the line
            frac = original_sequence_index / (self._original_coords_len - 1)
            interpolated_coord = (
                display_coords[0] * (1 - frac) + display_coords[1] * frac
            )
            return interpolated_coord

        # 5. Interpolate along the arrow axis (base midpoint to tip)
        # display_coords has shape [3, 3]: [left_base, right_base, tip]
        base_midpoint = (display_coords[0] + display_coords[1]) / 2.0
        tip_point = display_coords[2]

        # Calculate fraction along the length (0 = base midpoint, 1 = tip)
        # Based on position within the *original* sequence length
        frac = original_sequence_index / (self._original_coords_len - 1)

        # Linear interpolation between base midpoint and tip point
        interpolated_coord = base_midpoint * (1 - frac) + tip_point * frac

        return interpolated_coord

    def get_start_connection_point(self, structure: Structure) -> Optional[np.ndarray]:
        """Calculate the 2D coordinate for the start connection point.

        Args:
            structure: The core Structure object containing projected coordinates.

        Returns:
            A NumPy array [X, Y] or None if calculation fails.
        """
        coords_2d = self.get_coordinates(structure)[:, :2]
        if coords_2d is None:
            return None
        if len(coords_2d) < 3:
            return coords_2d[0, :2]

        return (coords_2d[0, :2] + coords_2d[1, :2]) / 2

    def get_end_connection_point(self, structure: Structure) -> Optional[np.ndarray]:
        """Calculate the 2D coordinate for the end connection point.

        Args:
            structure: The core Structure object containing projected coordinates.

        Returns:
            A NumPy array [X, Y] or None if calculation fails.
        """
        coords_2d = self.get_coordinates(structure)[:, :2]
        if coords_2d is None:
            return None
        return coords_2d[-1, :2]
