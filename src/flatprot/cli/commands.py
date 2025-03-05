# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for FlatProt visualization tool."""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console

from flatprot.cli.errors import (
    FlatProtCLIError,
    FileNotFoundError,
    InvalidStructureError,
)

console = Console()


def validate_structure_file(path: Path) -> None:
    """Validate that the file exists and is a valid PDB or CIF format.

    Args:
        path: Path to the structure file

    Raises:
        FileNotFoundError: If the file does not exist
        InvalidStructureError: If the file is not a valid PDB or CIF format
    """
    # Check file existence
    if not path.exists():
        raise FileNotFoundError(str(path))

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


def main(
    structure_file: Path,
    output_file: Path,
    matrix: Optional[Path] = None,
    annotations: Optional[Path] = None,
    style: Optional[Path] = None,
) -> None:
    """Generate a 2D visualization of a protein structure.

    Args:
        structure_file: Path to a PDB/CIF structure file to visualize.
        output_file: Path for the output SVG file containing the visualization.
        matrix: Path to a numpy matrix file for custom transformation. If not provided, inertia-based projection will be used.
        annotations: Path to a TOML file containing annotation specifications.
        style: Path to a TOML file containing style specifications.

    Examples:
        flatprot structure.pdb output.svg
        flatprot structure.cif output.svg --annotations annotations.toml --style style.toml
        flatprot structure.pdb output.svg --matrix custom_matrix.npy
    """
    try:
        # Validate the structure file
        validate_structure_file(structure_file)

        # Ensure the output directory exists
        output_dir = output_file.parent
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)

        # Verify optional files if specified
        if matrix and not matrix.exists():
            raise FileNotFoundError(str(matrix))
        if annotations and not annotations.exists():
            raise FileNotFoundError(str(annotations))
        if style and not style.exists():
            raise FileNotFoundError(str(style))

        # For now, just print a message to indicate successful parsing
        console.print("[bold green]Successfully parsed arguments:[/bold green]")
        console.print(f"  Structure file: {str(structure_file)}")
        console.print(f"  Output file: {str(output_file)}")
        if matrix:
            console.print(f"  Matrix file: {str(matrix)}")
        if annotations:
            console.print(f"  Annotations file: {str(annotations)}")
        if style:
            console.print(f"  Style file: {str(style)}")

        return 0

    except FlatProtCLIError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        return 1
