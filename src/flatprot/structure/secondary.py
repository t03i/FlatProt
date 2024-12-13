# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class SecondaryStructureType(Enum):
    HELIX = "H"
    SHEET = "S"
    COIL = "O"


class SecondaryStructure:
    def __init__(self, type: SecondaryStructureType, start: int, end: int):
        self.type = type
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.type.name} {self.start} {self.end}"


class Helix(SecondaryStructure):
    def __init__(self, start: int, end: int):
        super().__init__(SecondaryStructureType.HELIX, start, end)


class Sheet(SecondaryStructure):
    def __init__(self, start: int, end: int):
        super().__init__(SecondaryStructureType.SHEET, start, end)


class Coil(SecondaryStructure):
    def __init__(self, start: int, end: int):
        super().__init__(SecondaryStructureType.COIL, start, end)


def createSecondaryStructure(
    ss_type: SecondaryStructureType, start: int, end: int
) -> SecondaryStructure:
    if ss_type == SecondaryStructureType.HELIX:
        return Helix(start, end)
    elif ss_type == SecondaryStructureType.SHEET:
        return Sheet(start, end)
    else:
        return Coil(start, end)
