# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..core.structure import Structure

from .errors import StructureFileNotFoundError, InvalidStructureError


def validate_structure_file(path: Path) -> None:
    """Validate that the file exists and is a valid PDB or CIF format.

    Args:
        path: Path to the structure file

    Raises:
        StructureFileNotFoundError: If the file does not exist
        InvalidStructureError: If the file is not a valid PDB or CIF format
    """
    # Check file existence
    if not path.exists():
        raise StructureFileNotFoundError(str(path))

    # Check file extension
    suffix = path.suffix.lower()
    if suffix not in [".pdb", ".cif", ".mmcif", ".ent"]:
        raise InvalidStructureError(
            str(path),
            "PDB or CIF",
            "File does not have a recognized structure file extension (.pdb, .cif, .mmcif, .ent)",
        )

    # Basic content validation
    try:
        with open(path, "r") as f:
            content = f.read(1000)  # Read first 1000 bytes for quick check

            # Basic check for PDB format
            if suffix in [".pdb", ".ent"]:
                if not (
                    "ATOM" in content or "HETATM" in content or "HEADER" in content
                ):
                    raise InvalidStructureError(
                        str(path),
                        "PDB",
                        "File does not contain required PDB records (ATOM, HETATM, or HEADER)",
                    )

            # Basic check for mmCIF format
            if suffix in [".cif", ".mmcif"]:
                if not (
                    "_atom_site." in content or "loop_" in content or "data_" in content
                ):
                    raise InvalidStructureError(
                        str(path),
                        "CIF",
                        "File does not contain required CIF categories (_atom_site, loop_, or data_)",
                    )
    except UnicodeDecodeError:
        raise InvalidStructureError(
            str(path),
            "PDB or CIF",
            "File contains invalid characters and is not a valid text file",
        )


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
