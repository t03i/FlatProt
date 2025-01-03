# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0
from typing import Iterator

import numpy as np

from .residue import Residue
from .secondary import (
    SecondaryStructure,
    SecondaryStructureType,
    createSecondaryStructure,
)


class StructureComponent:
    """
    Base class for a structure component.
    """

    def __init__(
        self,
        residues: list[Residue],
        index: np.ndarray,
        coordinates: np.ndarray,
    ):
        self.__residues = residues
        self.__index = index
        self.__coordinates = coordinates

    @property
    def coordinates(self) -> np.ndarray:
        return self.__coordinates

    @property
    def residues(self) -> list[Residue]:
        return self.__residues

    @property
    def index(self) -> np.ndarray:
        return self.__index

    def get_coordinates_for_index(self, index: int) -> np.array:
        pos = (self.index == index).nonzero()[0]
        return self.coordinates[pos, :]


class Chain(StructureComponent):
    def __init__(
        self,
        chain_id: str,
        residues: list[Residue],
        index: np.ndarray,
        coordinates: np.ndarray,
    ):
        super().__init__(residues, index, coordinates)
        self.id = chain_id
        self.__secondary_structure: list[SecondaryStructure] = []

    def add_secondary_structure(
        self, type: SecondaryStructureType, atom_start: int, atom_end: int
    ) -> None:
        start_idx = np.where(self.index == atom_start)[0][0]
        end_idx = np.where(self.index == atom_end)[0][0]
        self.__secondary_structure.append(
            createSecondaryStructure(type, start_idx, end_idx)
        )

    @property
    def secondary_structure(self) -> list[SecondaryStructure]:
        # Sort existing secondary structure elements
        sorted_ss = sorted(self.__secondary_structure, key=lambda x: x.start)

        # Fill gaps with coils
        complete_ss = []
        current_pos = 0

        for ss in sorted_ss:
            # If there's a gap before this secondary structure, add a coil
            if ss.start > current_pos:
                complete_ss.append(
                    createSecondaryStructure(
                        SecondaryStructureType.COIL, current_pos, ss.start - 1
                    )
                )
            complete_ss.append(ss)
            current_pos = ss.end + 1

        # If there's space after the last secondary structure, fill with coil
        if current_pos < len(self.index):
            complete_ss.append(
                createSecondaryStructure(
                    SecondaryStructureType.COIL, current_pos, len(self.index) - 1
                )
            )

        return complete_ss

    def __str__(self) -> str:
        return f"Chain {self.id}"

    @property
    def num_residues(self) -> int:
        return len(self.index)

    def iter_secondary_structure_with_coordinates(self):
        """Iterate over all secondary structure elements with their coordinates.

        Yields:
            tuple: (secondary_structure_element, coordinates)
        """
        for element in self.secondary_structure:
            coords = self.coordinates[element.start : element.end + 1]
            yield element, coords

    def iter_secondary_structure_with_residues(self):
        """Iterate over all secondary structure elements with their residues.

        Yields:
            tuple: (secondary_structure_element, residues)
        """
        for element in self.secondary_structure:
            residues = self.residues[element.start : element.end + 1]
            yield element, residues

    def iter_secondary_structure_with_data(self):
        """Iterate over all secondary structure elements with their coordinates and residues.

        Yields:
            tuple: (secondary_structure_element, coordinates, residues)
        """
        for element in self.secondary_structure:
            coords = self.coordinates[element.start : element.end + 1]
            residues = self.residues[element.start : element.end + 1]
            yield element, coords, residues


class Structure:
    def __init__(self, chains: list[Chain]):
        self.__chains = {chain.id: chain for chain in chains}

    def __getitem__(self, chain_id: str) -> Chain:
        return self.__chains[chain_id]

    def __contains__(self, chain_id: str) -> bool:
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
    def residues(self) -> list[Residue]:
        """Get all residues from all chains concatenated in a single list."""
        all_residues = []
        for chain in self.__chains.values():
            all_residues.extend(chain.residues)
        return all_residues

    @property
    def coordinates(self) -> np.ndarray:
        """Get coordinates from all chains concatenated in a single array."""
        return np.concatenate([chain.coordinates for chain in self.__chains.values()])

    def iter_secondary_structure_with_data(self):
        """Iterate over all secondary structure elements with their coordinates.

        Yields:
            tuple: (secondary_structure_element, coordinates)
        """
        offset = 0
        for chain in self:
            for element, coords, residues in chain.iter_secondary_structure_with_data():
                yield element, coords, residues
                offset += len(coords)
