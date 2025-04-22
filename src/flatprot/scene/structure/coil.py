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


# Utility function (can be kept here or moved to a utils module)
def smooth_coordinates(coords: np.ndarray, reduction_factor: float = 0.2) -> np.ndarray:
    """Reduce point complexity using uniform selection.

    Args:
        coords: Input coordinates of shape (N, D) where D >= 2 (e.g., N, 2 or N, 3)
        reduction_factor: Fraction of points to keep (0.0-1.0)

    Returns:
        Simplified coordinates array (shape M, D) and the indices used.
    """
    n_points = len(coords)
    if n_points <= 3 or reduction_factor >= 1.0:
        # Return original coordinates and all indices
        return coords, np.arange(n_points)

    # Always keep first and last points
    # Ensure target_points is at least 2 if n_points >= 2
    target_points = (
        max(2, int(n_points * reduction_factor)) if n_points >= 2 else n_points
    )
    if target_points >= n_points:
        return coords, np.arange(n_points)

    # Use linear indexing for uniform point selection including start and end
    indices = np.linspace(0, n_points - 1, target_points, dtype=int)
    return coords[indices], indices


# --- Coil Specific Style ---
class CoilStyle(BaseStructureStyle):
    """Style properties specific to Coil elements."""

    # Override inherited defaults
    color: Color = Field(
        default=Color("#5b5859"),
        description="Default color for coil (light grey).",
    )
    stroke_width: float = Field(default=1.0, description="Line width for coil.")

    # Coil-specific attribute
    smoothing_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of points to keep during smoothing (0.0 to 1.0)."
        "Higher value means less smoothing.",
    )


# --- Coil Scene Element ---
class CoilSceneElement(BaseStructureSceneElement[CoilStyle]):
    """Represents a Coil segment of a protein structure.

    Renders as a smoothed line based on the pre-projected coordinates.
    """

    def __init__(
        self,
        residue_range_set: ResidueRangeSet,
        style: Optional[CoilStyle] = None,
        parent: Optional[SceneGroupType] = None,
    ):
        """Initializes the CoilSceneElement."""
        super().__init__(residue_range_set, style, parent)
        # Cache for the calculated smoothed coordinates and original indices
        self._cached_smoothed_coords: Optional[np.ndarray] = None
        self._original_indices: Optional[np.ndarray] = None
        self._original_coords_len: Optional[int] = None

    @property
    def default_style(self) -> CoilStyle:
        """Provides the default style for Coil elements."""
        return CoilStyle()

    def _get_original_coords_slice(self, structure: Structure) -> Optional[np.ndarray]:
        """Helper to extract the original coordinate slice for this coil."""
        coords_list = []
        if not self.residue_range_set.ranges:
            raise CoordinateCalculationError(
                f"Cannot get coordinates for Coil '{self.id}': no residue ranges defined."
            )
        coil_range = self.residue_range_set.ranges[0]

        try:
            chain = structure[coil_range.chain_id]
            for res_idx in range(coil_range.start, coil_range.end + 1):
                if res_idx in chain:
                    coord_idx = chain.coordinate_index(res_idx)
                    if 0 <= coord_idx < len(structure.coordinates):
                        coords_list.append(structure.coordinates[coord_idx])
                    else:
                        raise CoordinateCalculationError(
                            f"Coil '{self.id}': Coordinate index {coord_idx} out of bounds for residue {coil_range.chain_id}:{res_idx}."
                        )
                else:
                    raise CoordinateCalculationError(
                        f"Coil '{self.id}': Residue {coil_range.chain_id}:{res_idx} not found in chain coordinate map."
                    )
        except (KeyError, IndexError, AttributeError) as e:
            raise CoordinateCalculationError(
                f"Error getting original coordinates for Coil '{self.id}': {e}"
            ) from e

        return np.array(coords_list) if coords_list else None

    def get_coordinates(self, structure: Structure) -> Optional[np.ndarray]:
        """Retrieve the smoothed 2D + Depth coordinates for rendering the coil.

        Fetches the pre-projected coordinates from the structure, applies smoothing
        based on the style's smoothing_factor, and caches the result.

        Args:
            structure: The core Structure object containing pre-projected data.

        Returns:
            A NumPy array of smoothed 2D + Depth coordinates (X, Y, Depth).
        """
        # Return cached result if available
        if self._cached_smoothed_coords is not None:
            return self._cached_smoothed_coords

        # 1. Get the original (pre-projected) coordinates slice for this element
        original_coords = self._get_original_coords_slice(structure)
        if original_coords is None:
            self._cached_smoothed_coords = None
            self._original_indices = None
            self._original_coords_len = 0
            return None

        self._original_coords_len = len(original_coords)

        # Handle single-point coils separately
        if self._original_coords_len == 1:
            self._cached_smoothed_coords = original_coords
            self._original_indices = np.array([0])  # Index of the single point
            return self._cached_smoothed_coords

        # 2. Apply smoothing based on style (only if >= 2 points)
        smoothing_factor = self.style.smoothing_factor
        smoothed_coords, used_indices = smooth_coordinates(
            original_coords, smoothing_factor
        )

        # 3. Cache and return
        self._cached_smoothed_coords = smoothed_coords
        # Map the indices from smooth_coordinates (relative to the slice) back to the
        # original residue indices or coordinate indices if needed elsewhere, but
        # for get_2d_coordinate_at_residue, we primarily need the mapping *between*
        # original sequence index and smoothed sequence index.
        # We store the indices *within the original slice* that were kept.
        self._original_indices = used_indices

        return self._cached_smoothed_coords

    def get_coordinate_at_residue(
        self, residue: ResidueCoordinate, structure: Structure
    ) -> Optional[np.ndarray]:
        """Retrieves the specific 2D coordinate + Depth corresponding to a residue
        within the smoothed representation of the coil.

        Uses linear interpolation between the points of the smoothed coil line.

        Args:
            residue: The residue coordinate (chain and index) to find the 2D point for.
            structure: The core Structure object containing pre-projected 2D + Depth data.

        Returns:
            A NumPy array [X, Y, Depth] from the smoothed representation, potentially interpolated.
        """
        # 1. Ensure smoothed coordinates are calculated and cached
        # This call populates self._cached_smoothed_coords, self._original_coords_len, etc.
        smoothed_coords = self.get_coordinates(structure)
        if smoothed_coords is None or self._original_coords_len is None:
            return None  # Cannot determine coordinate if smoothing failed

        # 2. Check if residue is within the element's range
        if residue not in self.residue_range_set:
            return None
        # Assuming single range for simplicity
        element_range = self.residue_range_set.ranges[0]
        if residue.chain_id != element_range.chain_id:
            return None

        # 3. Map residue index to the 0-based index within the *original* sequence of this coil
        # This index represents the position *before* smoothing.
        try:
            original_sequence_index = residue.residue_index - element_range.start
            if not (0 <= original_sequence_index < self._original_coords_len):
                return None  # Residue index is outside the valid range for this element
        except Exception:
            return None  # Should not happen if residue is in range set, but defensive check

        # 4. Map the original sequence index to the fractional index within the *smoothed* sequence
        # This tells us where the original residue falls along the smoothed line.
        orig_len = self._original_coords_len
        smooth_len = len(smoothed_coords)

        # Avoid division by zero if original length was 1 (although checked earlier)
        if orig_len <= 1:
            return smoothed_coords[0] if smooth_len > 0 else None

        # Calculate fractional position along the smoothed line
        mapped_idx_frac = (original_sequence_index * (smooth_len - 1)) / (orig_len - 1)

        # 5. Linear interpolation between adjacent smoothed points
        idx_low = int(np.floor(mapped_idx_frac))
        # Clamp idx_high to the last valid index of the smoothed array
        idx_high = min(idx_low + 1, smooth_len - 1)
        # Ensure idx_low is also within bounds (handles edge case where mapped_idx_frac might be exactly smooth_len-1)
        idx_low = min(idx_low, smooth_len - 1)

        # Calculate interpolation fraction
        frac = mapped_idx_frac - idx_low

        # Interpolate X, Y, and Depth
        coord_low = smoothed_coords[idx_low]
        coord_high = smoothed_coords[idx_high]
        interpolated_coord = coord_low * (1 - frac) + coord_high * frac

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
        return coords_2d[0, :2]

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
