# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for FlatProt visualization tool."""

from pathlib import Path
from typing import Optional

from rich.console import Console

from flatprot.core.error import FlatProtError
from flatprot.cli.errors import error_handler
from flatprot.io import GemmiStructureParser
from flatprot.io import validate_structure_file, validate_optional_files
from flatprot.utils.transformation import create_coordinate_manager
from flatprot.utils.projection import apply_projection
from flatprot.utils.styling import create_style_manager
from flatprot.utils.scene import generate_svg, save_svg

console = Console()


def print_success_summary(
    structure_path: Path,
    output_path: Optional[Path],
    matrix_path: Optional[Path],
    style_path: Optional[Path],
    annotations_path: Optional[Path],
) -> None:
    """Print a summary of the successful operation.

    Args:
        structure_path: Path to the input structure file
        output_path: Path to the output SVG file, or None if printing to stdout
        matrix_path: Path to the custom transformation matrix file, or None if using default
        style_path: Path to the custom style file, or None if using default
        annotations_path: Path to the annotations file, or None if not using annotations
    """
    console.print("[bold green]Successfully processed structure:[/bold green]")
    console.print(f"  Structure file: {str(structure_path)}")
    console.print(f"  Output file: {str(output_path) if output_path else 'stdout'}")
    console.print(
        f"  Transformation: {'Custom matrix' if matrix_path else 'Inertia-based'}"
    )
    if matrix_path:
        console.print(f"  Matrix file: {str(matrix_path)}")
    if style_path:
        console.print(f"  Style file: {str(style_path)}")
    if annotations_path:
        console.print(f"  Annotations file: {str(annotations_path)}")


@error_handler
def main(
    structure: Path,
    output: Optional[Path] = None,
    matrix: Optional[Path] = None,
    style: Optional[Path] = None,
    annotations: Optional[Path] = None,
) -> int:
    """Generate a flat projection of a protein structure.

    Args:
        structure: Path to the structure file (PDB or similar).
        output: Path to save the SVG output.
            If not provided, the SVG is printed to stdout.
        matrix: Path to a custom transformation matrix.
            If not provided, a default inertia transformation is used.
        style: Path to a custom style file in TOML format.
            If not provided, the default styles are used.
        annotations: Path to a TOML file with annotation definitions.
            The annotation file can define point, line, and area annotations to highlight
            specific structural features. Examples:
            - Point annotations mark single residues with symbols
            - Line annotations connect residues with lines
            - Area annotations highlight regions of the structure
            See examples/annotations.toml for a reference annotation file.

    Returns:
        int: 0 for success, 1 for errors.

    Examples:
        flatprot structure.pdb output.svg
        flatprot structure.cif output.svg --annotations annotations.toml --style style.toml
        flatprot structure.pdb output.svg --matrix custom_matrix.npy
    """
    try:
        # Validate the structure file
        validate_structure_file(structure)

        # Verify optional files if specified
        validate_optional_files([matrix, style, annotations])

        # Load structure
        parser = GemmiStructureParser()
        structure_obj = parser.parse_structure(structure)

        # Create style manager
        style_manager = create_style_manager(style)

        # Process coordinates (transformation only)
        coordinate_manager = create_coordinate_manager(structure_obj, matrix)

        # Apply projection (using style information)
        coordinate_manager = apply_projection(coordinate_manager, style_manager)

        # Generate SVG visualization
        svg_content = generate_svg(
            structure_obj, coordinate_manager, style_manager, annotations
        )

        if output is not None:
            # Save SVG to output file
            save_svg(svg_content, output)
        else:
            # Print SVG to stdout
            console.print(svg_content)

        # Print success message
        print_success_summary(structure, output, matrix, style, annotations)

        return 0

    except FlatProtError as e:
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
        return 1
