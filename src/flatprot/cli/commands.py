# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for FlatProt visualization tool."""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console

from flatprot.cli.errors import FlatProtCLIError

console = Console()


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
    # Check that structure file exists
    if not structure_file.exists():
        raise FlatProtCLIError(f"Structure file not found: {structure_file}")

    # Ensure the output directory exists
    output_dir = output_file.parent
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    # Verify optional files if specified
    if matrix and not matrix.exists():
        raise FlatProtCLIError(f"Matrix file not found: {matrix}")
    if annotations and not annotations.exists():
        raise FlatProtCLIError(f"Annotations file not found: {annotations}")
    if style and not style.exists():
        raise FlatProtCLIError(f"Style file not found: {style}")

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
