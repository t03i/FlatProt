# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for FlatProt visualization tool.

This module provides the main entry point for the FlatProt CLI,
allowing users to generate 2D visualizations of protein structures
from the command line.

Examples:
    Basic usage:
        flatprot structure.pdb output.svg

    With custom styles and annotations:
        flatprot structure.cif output.svg --style styles.toml --annotations annotations.toml

    Using a custom transformation matrix:
        flatprot structure.pdb output.svg --matrix custom_matrix.npy

    Providing secondary structure information for PDB files:
        flatprot structure.pdb output.svg --dssp structure.dssp
"""

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
    """
    Print a summary of the successful operation.

    This function logs information about the processed files and options
    used during the visualization generation.

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
def project(
    structure: Path,
    output: Optional[Path] = None,
    matrix: Optional[Path] = None,
    style: Optional[Path] = None,
    annotations: Optional[Path] = None,
    dssp: Optional[Path] = None,
    quiet: Annotated[bool, Parameter(group=verbosity_group)] = False,
    verbose: Annotated[bool, Parameter(group=verbosity_group)] = False,
) -> int:
    """
    Generate a flat projection of a protein structure.

    This function is the main entry point for the FlatProt CLI. It processes
    a protein structure file, applies transformations, styles, and annotations,
    and generates a 2D SVG visualization.

    Args:
        structure: Path to the structure file (PDB or similar).
            Supported formats include PDB (.pdb) and mmCIF (.cif, .mmcif).
            The file must exist and be in a valid format.

        output: Path to save the SVG output.
            If not provided, the SVG is printed to stdout.
            The directory will be created if it doesn't exist.

        matrix: Path to a custom transformation matrix.
            If not provided, a default inertia transformation is used.
            The matrix should be a NumPy (.npy) file containing:
            - A 4x3 matrix where the first 3 rows are the rotation matrix (3x3)
              and the last row is the translation vector (1x3)
            - Alternatively, a 3x3 rotation matrix (translation will be set to zero)
            - The matrix can also be transposed (3x4) and will be automatically corrected

        style: Path to a custom style file in TOML format.
            If not provided, the default styles are used.
            The style file can define visual properties for helices, sheets,
            points, lines, and areas. See examples/styles.toml for reference.

        annotations: Path to a TOML file with annotation definitions.
            The annotation file can define point, line, and area annotations to highlight
            specific structural features. Examples:
            - Point annotations mark single residues with symbols
            - Line annotations connect residues with lines
            - Area annotations highlight regions of the structure
            See examples/annotations.toml for a reference annotation file.

        dssp: Path to a DSSP file with secondary structure assignments.
            If not provided, secondary structure is assumed to be in the input structure file.
            Required for PDB files, as they don't contain secondary structure information.

        quiet: Suppress all output except errors.
            When set, only error messages will be displayed.

        verbose: Print additional information.
            When set, debug-level messages will be displayed.

    Returns:
        int: 0 for success, 1 for errors.

    Examples:
        Basic usage:
            flatprot structure.pdb output.svg

        With custom styles and annotations:
            flatprot structure.cif output.svg --annotations annotations.toml --style style.toml

        Using a custom transformation matrix:
            flatprot structure.pdb output.svg --matrix custom_matrix.npy

        Providing secondary structure information for PDB files:
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


@error_handler
def align(
    structure_file: Path,
    output_file: Optional[Path] = None,
    matrix: Optional[Path] = None,
    foldseek_path: str = "foldseek",
    database_path: Optional[Path] = None,
    min_probability: float = 0.5,
    download_db: bool = False,
) -> int:
    """
    Align a protein structure to known superfamilies.

    This command finds the best matching protein superfamily for the given structure
    and returns alignment information including rotation matrices.

    Args:
        structure_file: Path to the input structure file (PDB or similar).
            Supported formats include PDB (.pdb) and mmCIF (.cif, .mmcif).
            The file must exist and be in a valid format.

        output_file: Path to save the JSON results.
            If not provided, the results are printed to stdout.
            The directory will be created if it doesn't exist.

        matrix: Path to a custom transformation matrix.
            If provided, this matrix will be applied to the structure before alignment.
            Otherwise, the default inertia-based transformation is used.

        foldseek_path: Path to the FoldSeek executable.
            Defaults to "foldseek" which assumes it's in the system PATH.
            Set this if FoldSeek is installed in a non-standard location.

        database_path: Path to a custom alignment database.
            If not provided, the default database will be used.
            If the database doesn't exist, it will be downloaded unless download_db is False.

        min_probability: Minimum probability threshold for alignments.
            Alignments with probability below this threshold will be rejected.
            Value should be between 0.0 and 1.0, with higher values being more stringent.

        download_db: Force database download even if it already exists.
            When set, the alignment database will be redownloaded regardless of
            whether it already exists locally.

    Returns:
        int: 0 for success, 1 for errors.

    Examples:
        Basic usage:
            flatprot align structure.pdb results.json

        Using a custom database:
            flatprot align structure.pdb results.json --database custom_db_path

        Adjusting probability threshold:
            flatprot align structure.pdb --min-probability 0.7
    """
    logger = logging.getLogger("flatprot.align")

    # Validate that the structure file exists
    if not structure_file.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")

    # Log the arguments
    logger.info(f"Structure file: {structure_file}")
    logger.info(f"Output file: {output_file or 'stdout'}")
    logger.info(f"FoldSeek path: {foldseek_path}")
    logger.info(f"Database path: {database_path or 'default'}")
    logger.info(f"Minimum probability: {min_probability}")
    logger.info(f"Force database download: {download_db}")

    # TODO: Implement alignment functionality in future prompts
    logger.info("Alignment functionality will be implemented in future steps")

    return 0
