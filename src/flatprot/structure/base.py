# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from .residue import Residue


class StructureComponent:
    """
    Base class for a structure component.
    """

    def __init__(self, residues: list[Residue], coordinates: np.ndarray):
        self.residues = residues
        self.coordinates = coordinates

    def get_coordinates(self) -> np.ndarray:
        return self.coordinates

    def get_residues(self) -> list[Residue]:
        return self.residues

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.coordinates = rotation @ self.coordinates + translation


class StructureComponentCollection(StructureComponent):
    def __init__(self, components: list[StructureComponent]):
        self.components = components

    def get_coordinates(self) -> np.ndarray:
        return np.concatenate(
            [component.get_coordinates() for component in self.components]
        )

    def get_residues(self) -> list[Residue]:
        return [
            residue
            for component in self.components
            for residue in component.get_residues()
        ]

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        for component in self.components:
            component.transform(rotation, translation)
