# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0
from typing import Iterator, Dict, List, Callable, Optional

import numpy as np

from .types import ResidueType, SecondaryStructureType
from .coordinates import ResidueRange, ResidueCoordinate


class Chain:
    def __init__(
        self,
        chain_id: str,
        residues: list[ResidueType],
        index: np.ndarray,
        coordinates: np.ndarray,
        secondary_structure: Optional[List[ResidueRange]] = None,
    ):
        """
        Initializes a protein Chain.

        Args:
            chain_id: The unique identifier for the chain (e.g., 'A').
            residues: A list of residue types corresponding to the coordinates.
            index: A NumPy array of residue indices (sequence numbers).
            coordinates: A NumPy array of shape (N, 3) containing C-alpha coordinates.
            secondary_structure: Optional list of predefined secondary structures.
        """
        if not (len(index) == len(residues) == len(coordinates)):
            raise ValueError(
                f"Index, residues, and coordinates must have the same length, "
                f"got {len(index)}, {len(residues)}, and {len(coordinates)} for chain {chain_id}"
            )

        self.id = chain_id
        self.__secondary_structure: List[ResidueRange] = secondary_structure or []
        self.__chain_coordinates: Dict[int, ResidueCoordinate] = {}
        # Store index directly for easier reconstruction
        self.__index = index
        for i, (idx, residue) in enumerate(zip(index, residues)):
            # Use 'i' as the coordinate_index (contiguous 0-based index)
            self.__chain_coordinates[idx] = ResidueCoordinate(chain_id, idx, residue, i)
        self.__coordinates = coordinates

    def __len__(self) -> int:
        return len(self.__chain_coordinates)

    @property
    def coordinates(self) -> np.ndarray:
        return self.__coordinates

    @property
    def residues(self) -> list[ResidueType]:
        return [
            self.__chain_coordinates[idx].residue_type
            for idx in self.__chain_coordinates
        ]

    def _check_range_contiguous(
        self, residue_start: ResidueCoordinate, residue_end: ResidueCoordinate
    ) -> bool:
        """Check if all residues in the given range are present in the chain coordinates.

        Args:
            residue_start: The starting residue coordinate of the range
            residue_end: The ending residue coordinate of the range

        Returns:
            bool: True if all residues in the range are present in the chain coordinates,
                  False otherwise
        """
        # Ensure both residues are from this chain
        if residue_start.chain_id != self.id or residue_end.chain_id != self.id:
            return False

        # Check if all indices in the range are present
        for idx in range(residue_start.residue_index, residue_end.residue_index + 1):
            if idx not in self.__chain_coordinates:
                return False

        return True

    def add_secondary_structure(
        self,
        type: SecondaryStructureType,
        residue_start: int,
        residue_end: int,
    ) -> None:
        start_coord = self.__chain_coordinates.get(residue_start, None)
        end_coord = self.__chain_coordinates.get(residue_end, None)

        if start_coord is None or end_coord is None:
            raise ValueError(
                f"Residue {residue_start} or {residue_end} not found in chain {self.id}"
            )

        if not self._check_range_contiguous(start_coord, end_coord):
            raise ValueError(
                f"Residue range {residue_start} to {residue_end} is not contiguous"
            )

        self.__secondary_structure.append(
            ResidueRange(
                self.id,
                residue_start,
                residue_end,
                start_coord.coordinate_index,
                type,
            )
        )

    @property
    def secondary_structure(self) -> list[ResidueRange]:
        """Get all secondary structure elements, filling gaps with coils.

        Handles discontinuities in residue indexing by creating separate ranges
        for contiguous segments.

        Returns:
            list[ResidueRange]: A complete list of secondary structure elements
                covering all residues in the chain.
        """
        # Get all residue indices present in the chain, sorted
        residue_indices = sorted(self.__chain_coordinates.keys())

        if not residue_indices:
            return []

        # Create a map of residue index to defined secondary structure type
        # Sorting __secondary_structure ensures predictable behavior in case of overlaps (though overlaps shouldn't occur)
        defined_ss_map: Dict[int, SecondaryStructureType] = {}
        for ss_element in sorted(self.__secondary_structure):
            # Only consider residues actually present in the chain for this element
            for idx in range(ss_element.start, ss_element.end + 1):
                if idx in self.__chain_coordinates:
                    defined_ss_map[idx] = ss_element.secondary_structure_type

        complete_ss: List[ResidueRange] = []
        segment_start_idx = residue_indices[0]
        segment_type = defined_ss_map.get(
            segment_start_idx, SecondaryStructureType.COIL
        )
        segment_contig_start = self.__chain_coordinates[
            segment_start_idx
        ].coordinate_index

        # Iterate through residues to identify segments of continuous type and index
        for i in range(1, len(residue_indices)):
            current_idx = residue_indices[i]
            prev_idx = residue_indices[i - 1]
            current_res_type = defined_ss_map.get(
                current_idx, SecondaryStructureType.COIL
            )

            # Check for discontinuity in residue index or change in SS type
            if current_idx != prev_idx + 1 or current_res_type != segment_type:
                # End the previous segment
                segment_end_idx = prev_idx
                complete_ss.append(
                    ResidueRange(
                        self.id,
                        segment_start_idx,
                        segment_end_idx,
                        segment_contig_start,
                        segment_type,
                    )
                )

                # Start a new segment
                segment_start_idx = current_idx
                segment_type = current_res_type
                segment_contig_start = self.__chain_coordinates[
                    segment_start_idx
                ].coordinate_index

        # Add the final segment
        segment_end_idx = residue_indices[-1]
        complete_ss.append(
            ResidueRange(
                self.id,
                segment_start_idx,
                segment_end_idx,
                segment_contig_start,
                segment_type,
            )
        )
        return complete_ss

    def __str__(self) -> str:
        return f"Chain {self.id}"

    @property
    def num_residues(self) -> int:
        return len(self.__chain_coordinates)

    def to_ranges(self) -> list[ResidueRange]:
        """Convert chain to a list of ResidueRange objects.

        Handles non-contiguous residue indices by creating separate ranges
        for each contiguous segment.

        Returns:
            list[ResidueRange]: List of residue ranges representing this chain
        """
        if self.index is None or len(self.index) == 0:
            return []

        ranges = []
        start_idx = 0

        # Iterate through indices to find breaks in continuity
        for i in range(1, len(self.index)):
            # If there's a gap in the indices, end the current range and start a new one
            if self.index[i] != self.index[i - 1] + 1:
                ranges.append(
                    ResidueRange(
                        self.id,
                        self.index[start_idx],
                        self.index[i - 1],
                        self.__chain_coordinates[
                            self.index[start_idx]
                        ].coordinate_index,
                    )
                )
                start_idx = i

        # Add the final range
        ranges.append(
            ResidueRange(
                self.id,
                self.index[start_idx],
                self.index[-1],
                self.__chain_coordinates[self.index[start_idx]].coordinate_index,
            )
        )

        return ranges

    def __iter__(self) -> Iterator[ResidueCoordinate]:
        return iter(self.__chain_coordinates.values())

    def __getitem__(self, residue_index: int) -> ResidueCoordinate:
        return self.__chain_coordinates[residue_index]

    def __contains__(self, residue_index: int) -> bool:
        return residue_index in self.__chain_coordinates

    def coordinate_index(self, residue_index: int) -> int:
        return self.__chain_coordinates[residue_index].coordinate_index

    @property
    def index(self) -> np.ndarray:
        """Return the original residue index array."""
        return self.__index

    def apply_vectorized_transformation(
        self, transformer_func: Callable[[np.ndarray], np.ndarray]
    ) -> "Chain":
        """
        Applies a transformation function to the coordinates and returns a new Chain.

        The transformation function operates on the entire coordinate array of the chain.

        Args:
            transformer_func: A function that takes an Nx3 numpy array of coordinates
                              and returns a new Nx3 numpy array of transformed coordinates.

        Returns:
            A new Chain instance with the transformed coordinates, preserving all other
            metadata (ID, residues, indices, secondary structure).
        """
        new_coordinates = transformer_func(self.coordinates)
        if new_coordinates.shape != self.coordinates.shape:
            raise ValueError(
                f"Transformer function changed coordinate array shape "
                f"from {self.coordinates.shape} to {new_coordinates.shape}"
            )

        # Create a new chain instance with the new coordinates
        # Need to pass residues and index from the original chain
        new_chain = Chain(
            chain_id=self.id,
            residues=self.residues,  # Use property to get list
            index=self.index,  # Use stored index
            coordinates=new_coordinates,
            secondary_structure=list(self.__secondary_structure),  # Pass a copy
        )
        # Note: Secondary structure is shallow copied. If ResidueRange becomes mutable, deepcopy might be needed.

        return new_chain


class Structure:
    """Represents a complete protein structure composed of multiple chains."""

    def __init__(self, chains: list[Chain], id: Optional[str] = None):
        """Initializes a Structure object.

        Args:
            chains: A list of Chain objects representing the chains in the structure.
            id: An optional identifier for the structure (e.g., PDB ID or filename stem).
        """
        self.__chains = {chain.id: chain for chain in chains}
        self.id = id or "unknown_structure"  # Assign ID or a default
        # Pre-calculate total coordinates for validation if needed
        self._total_coordinates = sum(
            len(chain.coordinates) if chain.coordinates is not None else 0
            for chain in self.values()
        )

    def __getitem__(self, chain_id: str) -> Chain:
        """Get a chain by its ID."""
        return self.__chains[chain_id]

    def __contains__(self, chain_id: str | ResidueCoordinate) -> bool:
        """Check if a chain ID or ResidueCoordinate exists in the structure."""
        if isinstance(chain_id, str):
            return chain_id in self.__chains
        elif isinstance(chain_id, ResidueCoordinate):
            return (
                chain_id.chain_id in self.__chains
                and chain_id.residue_index in self.__chains[chain_id.chain_id]
            )
        return False

    def __iter__(self) -> Iterator[tuple[str, Chain]]:
        """Iterate over chain IDs and Chain objects."""
        return iter(self.__chains.items())

    def items(self) -> Iterator[tuple[str, Chain]]:
        """Return an iterator over chain ID / Chain pairs."""
        return self.__chains.items()

    def values(self) -> Iterator[Chain]:
        """Return an iterator over Chain objects."""
        return self.__chains.values()

    def __len__(self) -> int:
        """Return the number of chains in the structure."""
        return len(self.__chains)

    @property
    def residues(self) -> list[ResidueType]:
        """Get a flattened list of all residues across all chains."""
        all_residues = []
        for chain in self.__chains.values():
            all_residues.extend(chain.residues)
        return all_residues

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        """Get a concatenated array of all coordinates across all chains, or None if empty."""
        all_coords = [
            chain.coordinates
            for chain in self.__chains.values()
            if chain.coordinates is not None and chain.coordinates.size > 0
        ]
        if not all_coords:
            return None
        return np.vstack(all_coords)

    def __str__(self) -> str:
        return f"Structure(ID: {self.id}, Chains: {list(self.__chains.keys())})"

    def apply_vectorized_transformation(
        self, transformer_func: Callable[[np.ndarray], np.ndarray]
    ) -> "Structure":
        """Applies a transformation function to all coordinates and returns a new Structure.

        Args:
            transformer_func: A function that takes an (N, 3) coordinate array
                              and returns a transformed (N, 3) array.

        Returns:
            A new Structure instance with transformed coordinates.
        """
        new_chains = []
        start_index = 0
        original_coords = self.coordinates
        if original_coords is None:
            # Handle case with no coordinates - return a structure with chains having None coordinates
            # Recreate chains with None coords, ensuring topology is preserved
            for _, chain in self.items():
                # Create new chain with None coords
                new_chains.append(
                    Chain(
                        chain_id=chain.id,
                        residues=chain.residues,  # Use property to get list
                        index=chain.index,  # Use original index
                        coordinates=None,  # Explicitly None
                        # Pass a copy of the original secondary structure list
                        secondary_structure=list(chain._Chain__secondary_structure),
                    )
                )
            return Structure(new_chains, id=self.id)

        transformed_coords = transformer_func(original_coords)

        # Check for shape change after transformation
        if transformed_coords.shape != original_coords.shape:
            raise ValueError(
                f"Transformer function changed coordinate array shape from "
                f"{original_coords.shape} to {transformed_coords.shape}"
            )

        for (
            _,
            chain,
        ) in (
            self.items()
        ):  # Iterate in insertion order (Python 3.7+) or sorted order if needed
            num_coords_in_chain = (
                len(chain.coordinates) if chain.coordinates is not None else 0
            )
            if num_coords_in_chain > 0:
                # Slice the transformed coordinates for the current chain
                chain_transformed_coords = transformed_coords[
                    start_index : start_index + num_coords_in_chain
                ]
                # Create the new chain with the sliced coordinates
                new_chains.append(
                    Chain(
                        chain_id=chain.id,
                        residues=chain.residues,
                        index=chain.index,
                        coordinates=chain_transformed_coords,
                        # Pass a copy of the original secondary structure list
                        secondary_structure=list(chain._Chain__secondary_structure),
                    )
                )
                start_index += num_coords_in_chain
            else:
                # Handle chains originally having no coordinates
                new_chains.append(
                    Chain(
                        chain_id=chain.id,
                        residues=chain.residues,
                        index=chain.index,
                        coordinates=None,  # Keep coordinates as None
                        secondary_structure=list(chain._Chain__secondary_structure),
                    )
                )

        # Ensure all coordinates were assigned
        if start_index != transformed_coords.shape[0]:
            raise ValueError(
                f"Coordinate slicing error: processed {start_index} coordinates, "
                f"but expected {transformed_coords.shape[0]}."
            )

        return Structure(new_chains, id=self.id)

    def get_coordinate_at_residue(
        self, residue: ResidueCoordinate
    ) -> Optional[np.ndarray]:
        """Get the 3D coordinate for a specific residue.

        Args:
            residue: The residue coordinate to query.

        Returns:
            A NumPy array of coordinates (shape [3]) representing the residue's
            position (X, Y, Z), or None if the residue is not found.
        """
        chain = self.__chains.get(residue.chain_id)
        if chain is None:
            return None

        # Use the Chain's __contains__ and __getitem__ for residue lookup
        if residue.residue_index in chain:
            target_residue_coord = chain[residue.residue_index]
            # Access the coordinates using the coordinate_index stored in ResidueCoordinate
            # Ensure the chain has coordinates and the index is valid
            if (
                chain.coordinates is not None
                and 0 <= target_residue_coord.coordinate_index < len(chain.coordinates)
            ):
                return chain.coordinates[target_residue_coord.coordinate_index]

        # Residue index not found in chain or coordinate index invalid
        return None

    def with_coordinates(self, coordinates: np.ndarray) -> "Structure":
        """Create a new Structure with the given coordinates, preserving topology.

        Args:
            coordinates: A NumPy array of shape (N, 3) containing the new coordinates,
                         ordered consistently with the original structure's concatenated coordinates.

        Returns:
            A new Structure instance with the provided coordinates.

        Raises:
            ValueError: If the shape or total number of input coordinates does not match
                        the original structure's coordinate count.
        """
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("Input coordinates must be a numpy array.")
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                f"Input coordinates must have shape (N, 3), got {coordinates.shape}"
            )

        # Validate that the number of provided coordinates matches the original structure
        if coordinates.shape[0] != self._total_coordinates:
            raise ValueError(
                f"Input coordinates count ({coordinates.shape[0]}) does not match "
                f"the original structure's total coordinates ({self._total_coordinates})."
            )

        new_chains = []
        start_index = 0
        for (
            _,
            chain,
        ) in self.items():  # Iterate through original chains to maintain order
            num_coords_in_chain = (
                len(chain.coordinates) if chain.coordinates is not None else 0
            )
            if num_coords_in_chain > 0:
                # Slice the *new* coordinates array for the current chain
                new_chain_coords = coordinates[
                    start_index : start_index + num_coords_in_chain
                ]
                # Create the new chain with the sliced coordinates
                new_chains.append(
                    Chain(
                        chain_id=chain.id,
                        residues=chain.residues,
                        index=chain.index,
                        coordinates=new_chain_coords,
                        # Pass a copy of the original secondary structure list
                        secondary_structure=list(chain._Chain__secondary_structure),
                    )
                )
                start_index += num_coords_in_chain
            else:
                # Handle chains originally having no coordinates
                new_chains.append(
                    Chain(
                        chain_id=chain.id,
                        residues=chain.residues,
                        index=chain.index,
                        coordinates=None,  # Keep coordinates as None
                        secondary_structure=list(chain._Chain__secondary_structure),
                    )
                )

        # Final check to ensure all provided coordinates were used
        if start_index != coordinates.shape[0]:
            raise ValueError(  # Should not happen if initial count check passes, but good sanity check
                f"Coordinate assignment error during 'with_coordinates': processed {start_index} coordinates, "
                f"but input had {coordinates.shape[0]}."
            )

        return Structure(new_chains, id=self.id)
