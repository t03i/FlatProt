# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..structure.components import Structure


class StructureParser(ABC):
    """Abstract interface for parsing protein structure files."""

    @abstractmethod
    def parse_structure(
        self, structure_file: Path, secondary_structure_file: Optional[Path] = None
    ) -> Structure:
        """Parse a structure file into a dictionary of Protein models.

        Args:
            structure_file: Path to the structure file (e.g., PDB file)
            secondary_structure_file: Path to the secondary structure file (e.g., DSSP file), if not specified, secondary structure is assumed to be in the structure file
        Returns:
            Dictionary mapping chain IDs to Protein objects
        """
        pass

    def save_structure(
        self,
        structure: Structure,
        output_file: Path,
        separate_chains: bool = False,
    ) -> None:
        """Save a dictionary of Protein objects to a file.

        Args:
            structure: Dictionary mapping chain IDs to Protein objects
            output_file: Path to the output file
            separate_chains: If True, save each chain to a separate file
        """
        pass
