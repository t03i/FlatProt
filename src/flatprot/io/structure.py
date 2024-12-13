# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from ..structure.components import Protein


class StructureParser(ABC):
    """Abstract interface for parsing protein structure files."""

    @abstractmethod
    def parse_structure(self, structure_file: Path) -> dict[str, Protein]:
        """Parse a structure file into a dictionary of Protein models.

        Args:
            structure_file: Path to the structure file (e.g., PDB file)

        Returns:
            Dictionary mapping chain IDs to Protein objects
        """
        pass

    def save_structure(self, structure: dict[str, Protein], output_file: Path) -> None:
        """Save a dictionary of Protein objects to a file.

        Args:
            structure: Dictionary mapping chain IDs to Protein objects
            output_file: Path to the output file
        """
        pass
