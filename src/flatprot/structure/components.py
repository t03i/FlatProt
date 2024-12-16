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


class Chain:
    def __init__(
        self,
        chain_id: str,
        residues: list[Residue],
        index: np.ndarray,
        coordinates: np.ndarray,
    ):
        self.id = chain_id
        self.structure_component = StructureComponent(residues, index, coordinates)
        self.__secondary_structure: list[SecondaryStructure] = []

    def add_secondary_structure(
        self, type: SecondaryStructureType, atom_start: int, atom_end: int
    ) -> None:
        start_idx = np.where(self.structure_component.index == atom_start)[0][0]
        end_idx = np.where(self.structure_component.index == atom_end)[0][0]
        self.__secondary_structure.append(
            createSecondaryStructure(type, start_idx, end_idx)
        )

    @property
    def secondary_structure(self) -> list[SecondaryStructure]:
        return sorted(self.__secondary_structure, key=lambda x: x.start)

    def __str__(self) -> str:
        return f"Chain {self.id}"

    @property
    def num_residues(self) -> int:
        return len(self.structure_component.index)


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

    def __len__(self) -> int:
        return len(self.__chains)
