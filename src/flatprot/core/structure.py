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
                    defined_ss_map[idx] = ss_element.secondary_structure

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
    def __init__(self, chains: list[Chain]):
        """Initializes a Structure from a list of Chains."""
        self.__chains = {chain.id: chain for chain in chains}

    def __getitem__(self, chain_id: str) -> Chain:
        return self.__chains[chain_id]

    def __contains__(self, chain_id: str | ResidueCoordinate) -> bool:
        if isinstance(chain_id, ResidueCoordinate):
            return (
                chain_id.chain_id in self.__chains
                and chain_id.residue_index in self.__chains[chain_id.chain_id]
            )
        return chain_id in self.__chains

    def __iter__(self) -> Iterator[Chain]:
        return iter(self.__chains.values())

    def items(self) -> Iterator[tuple[str, Chain]]:
        return self.__chains.items()

    def values(self) -> Iterator[Chain]:
        return self.__chains.values()

    def __len__(self) -> int:
        return len(self.__chains)

    @property
    def residues(self) -> list[ResidueType]:
        """Get all residues from all chains concatenated in a single list."""
        all_residues = []
        for chain in self.__chains.values():
            all_residues.extend(chain.residues)
        return all_residues

    @property
    def coordinates(self) -> np.ndarray:
        """Get coordinates from all chains concatenated in a single array."""
        return np.concatenate([chain.coordinates for chain in self.__chains.values()])

    def __str__(self) -> str:
        return f"Structure with {len(self.__chains)} chains"

    def apply_vectorized_transformation(
        self, transformer_func: Callable[[np.ndarray], np.ndarray]
    ) -> "Structure":
        """
        Applies a transformation function to the combined coordinates of all chains
        and returns a new Structure.

        The transformation function operates on the concatenated coordinate array
        of the entire structure.

        Args:
            transformer_func: A function that takes an Nx3 numpy array of coordinates
                              (where N is the total number of residues in the structure)
                              and returns a new Nx3 numpy array of transformed coordinates.

        Returns:
            A new Structure instance containing new Chain instances with the
            transformed coordinates, preserving all other metadata.
        """
        original_coordinates = self.coordinates
        transformed_coordinates = transformer_func(original_coordinates)

        if transformed_coordinates.shape[0] != original_coordinates.shape[0]:
            raise ValueError(
                f"Transformer function changed coordinate array shape from "
                f"{original_coordinates.shape} to {transformed_coordinates.shape}"
            )

        new_chains = []
        start_idx = 0
        for chain in self.values():
            num_coords = len(chain.coordinates)
            end_idx = start_idx + num_coords
            chain_coords_transformed = transformed_coordinates[start_idx:end_idx]

            # Use the Chain's own transformation logic internally by creating
            # a lambda that just returns the pre-transformed segment.
            # This reuses the Chain's __init__ logic correctly.
            # Alternatively, manually create the new Chain like below:
            new_chain = Chain(
                chain_id=chain.id,
                residues=chain.residues,
                index=chain.index,
                coordinates=chain_coords_transformed,
                secondary_structure=list(chain.secondary_structure),  # Pass copy
            )
            new_chains.append(new_chain)
            start_idx = end_idx

        return Structure(new_chains)
