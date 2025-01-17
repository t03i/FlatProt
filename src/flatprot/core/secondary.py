# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import numpy as np

from flatprot.core.components import Residue


class SecondaryStructureType(Enum):
    HELIX = "H"
    SHEET = "S"
    COIL = "O"


class SecondaryStructure:
    def __init__(
        self,
        type: SecondaryStructureType,
        start: int,
        end: int,
        coordinates: np.ndarray,
        residue_indices: np.ndarray,
        residues: list[Residue],
    ):
        self.type = type
        self.start = start
        self.end = end
        # Create read-only views for both arrays
        self.__residue_indices = residue_indices
        self.__coordinates = coordinates
        self.__residues = residues

        # Make both views immutable
        self.__residue_indices.flags.writeable = False
        self.__coordinates.flags.writeable = False

    def __str__(self):
        return f"{self.type.name} {self.start} {self.end}"

    def __repr__(self):
        return f"{self.type.name} {self.start} {self.end}"

    def __len__(self):
        return self.end - self.start

    @property
    def coordinates(self) -> np.ndarray:
        return self.__coordinates

    def get_coordinates_for_residue(self, residue_idx: int) -> np.ndarray:
        """Get coordinates for a specific residue index."""
        mask = self.__residue_indices == residue_idx
        if not np.any(mask):
            raise ValueError(f"Residue index {residue_idx} not found in this structure")
        return self.__coordinates[mask]

    @property
    def residue_indices(self) -> np.ndarray:
        """Get the residue indices for this structure element."""
        return self.__residue_indices


class Helix(SecondaryStructure):
    def __init__(
        self,
        start: int,
        end: int,
        coordinates: np.ndarray,
        residue_indices: np.ndarray,
        residues: np.ndarray,
    ):
        super().__init__(
            SecondaryStructureType.HELIX,
            start,
            end,
            coordinates,
            residue_indices,
            residues,
        )


class Sheet(SecondaryStructure):
    def __init__(
        self,
        start: int,
        end: int,
        coordinates: np.ndarray,
        residue_indices: np.ndarray,
        residues: np.ndarray,
    ):
        super().__init__(
            SecondaryStructureType.SHEET,
            start,
            end,
            coordinates,
            residue_indices,
            residues,
        )


class Coil(SecondaryStructure):
    def __init__(
        self,
        start: int,
        end: int,
        coordinates: np.ndarray,
        residue_indices: np.ndarray,
        residues: np.ndarray,
    ):
        super().__init__(
            SecondaryStructureType.COIL,
            start,
            end,
            coordinates,
            residue_indices,
            residues,
        )


def createSecondaryStructure(
    ss_type: SecondaryStructureType,
    start: int,
    end: int,
    coordinates: np.ndarray,
    residue_indices: np.ndarray,
    residues: list[Residue],
) -> SecondaryStructure:
    if ss_type == SecondaryStructureType.HELIX:
        return Helix(start, end, coordinates, residue_indices, residues)
    elif ss_type == SecondaryStructureType.SHEET:
        return Sheet(start, end, coordinates, residue_indices, residues)
    else:
        return Coil(start, end, coordinates, residue_indices, residues)
