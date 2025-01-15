# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

import numpy as np


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
    ):
        self.type = type
        self.start = start
        self.end = end
        # Create read-only view of coordinates
        self.__coordinates = coordinates.view()
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


class Helix(SecondaryStructure):
    def __init__(self, start: int, end: int, coordinates: np.ndarray):
        super().__init__(SecondaryStructureType.HELIX, start, end, coordinates)


class Sheet(SecondaryStructure):
    def __init__(self, start: int, end: int, coordinates: np.ndarray):
        super().__init__(SecondaryStructureType.SHEET, start, end, coordinates)


class Coil(SecondaryStructure):
    def __init__(self, start: int, end: int, coordinates: np.ndarray):
        super().__init__(SecondaryStructureType.COIL, start, end, coordinates)


def createSecondaryStructure(
    ss_type: SecondaryStructureType,
    start: int,
    end: int,
    coordinates: np.ndarray,
) -> SecondaryStructure:
    if ss_type == SecondaryStructureType.HELIX:
        return Helix(start, end, coordinates)
    elif ss_type == SecondaryStructureType.SHEET:
        return Sheet(start, end, coordinates)
    else:
        return Coil(start, end, coordinates)
