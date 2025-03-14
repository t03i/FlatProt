# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for FlatProt visualization tool."""

from pathlib import Path
from typing import Optional, Annotated
import logging

from cyclopts import Parameter, Group, validators

from flatprot.core.error import FlatProtError
from flatprot.cli.errors import error_handler
from flatprot.io import GemmiStructureParser
from flatprot.io import validate_structure_file, validate_optional_files

from flatprot.utils import logger, setup_logging
from flatprot.utils.coordinate_manger import create_coordinate_manager, apply_projection
from flatprot.utils.svg import generate_svg, save_svg
from flatprot.utils.style import create_style_manager


def print_success_summary(
    structure_path: Path,
    output_path: Optional[Path],
    matrix_path: Optional[Path],
    style_path: Optional[Path],
    annotations_path: Optional[Path],
    dssp_path: Optional[Path],
) -> None:
    """Print a summary of the successful operation.

    Args:
        structure_path: Path to the input structure file
        output_path: Path to the output SVG file, or None if printing to stdout
        matrix_path: Path to the custom transformation matrix file, or None if using default
        style_path: Path to the custom style file, or None if using default
        annotations_path: Path to the annotations file, or None if not using annotations
        dssp_path: Path to the DSSP file for secondary structure information, or None if using default
    """
    logger.info("[bold]Successfully processed structure:[/bold]")
    logger.info(f"  Structure file: {str(structure_path)}")
    logger.info(f"  Output file: {str(output_path) if output_path else 'stdout'}")
    logger.info(
        f"  Transformation: {'Custom matrix' if matrix_path else 'Inertia-based'}"
    )
    if matrix_path:
        logger.info(f"  Matrix file: {str(matrix_path)}")
    if style_path:
        logger.info(f"  Style file: {str(style_path)}")
    if annotations_path:
        logger.info(f"  Annotations file: {str(annotations_path)}")
    if dssp_path:
        logger.info(f"  DSSP file: {str(dssp_path)}")


verbosity_group = Group(
    "Verbosity",
    default_parameter=Parameter(negative=""),  # Disable "--no-" flags
    validator=validators.MutuallyExclusive(),  # Only one option is allowed to be selected.
)


@error_handler
def main(
    structure: Path,
    output: Optional[Path] = None,
    matrix: Optional[Path] = None,
    style: Optional[Path] = None,
    annotations: Optional[Path] = None,
    dssp: Optional[Path] = None,
    quiet: Annotated[bool, Parameter(group=verbosity_group)] = False,
    verbose: Annotated[bool, Parameter(group=verbosity_group)] = False,
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
        dssp: Path to a DSSP file with secondary structure assignments.
            If not provided, secondary structure is assumed to be in the input structure file.
        quiet: Suppress all output except errors
        verbose: Print additional information

    Returns:
        int: 0 for success, 1 for errors.

    Examples:
        flatprot structure.pdb output.svg
        flatprot structure.cif output.svg --annotations annotations.toml --style style.toml
        flatprot structure.pdb output.svg --matrix custom_matrix.npy
        flatprot structure.pdb output.svg --dssp structure.dssp
    """

    # Set logging level based on verbosity flags

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    setup_logging(level)

    try:
        # Validate the structure file
        validate_structure_file(structure)

        # Check if secondary structure information can be extracted
        is_cif_file = structure.suffix.lower() in (".cif", ".mmcif")
        if not is_cif_file and dssp is None:
            raise FlatProtError(
                "Secondary structure information cannot be extracted from non-CIF files. "
                "Please provide either:\n"
                "  - A structure file in CIF format (.cif, .mmcif), or\n"
                "  - A DSSP file using the --dssp option\n"
                "Example: flatprot structure.pdb output.svg --dssp structure.dssp"
            )

        # Verify optional files if specified
        validate_optional_files([matrix, style, annotations, dssp])

        # Load structure
        parser = GemmiStructureParser()
        structure_obj = parser.parse_structure(structure, dssp)

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
            print(svg_content)

        # Print success message using the success method (which uses INFO level)
        print_success_summary(structure, output, matrix, style, annotations, dssp)

        return 0

    except FlatProtError as e:
        logger.error(e.message)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1
