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
from dataclasses import dataclass
import logging
import shutil
import subprocess

from cyclopts import Parameter, Group, validators

from flatprot.core.error import FlatProtError
from flatprot.cli.errors import error_handler
from flatprot.io import GemmiStructureParser
from flatprot.io import validate_structure_file, validate_optional_files

from flatprot.utils import logger, setup_logging
from flatprot.utils.coordinate_manger import create_coordinate_manager, apply_projection
from flatprot.utils.svg import generate_svg, save_svg
from flatprot.utils.style import create_style_manager
from flatprot.utils.database import ensure_database_available

from flatprot.alignment import align_structure_database, get_aligned_rotation_database
from flatprot.alignment import AlignmentDatabase
from flatprot.alignment import (
    NoSignificantAlignmentError,
    DatabaseEntryNotFoundError,
)
from flatprot.io import OutputFileError, InvalidStructureError
from flatprot.utils.alignment import save_alignment_results, save_alignment_matrix

verbosity_group = Group(
    "Verbosity",
    default_parameter=Parameter(negative=""),  # Disable "--no-" flags
    validator=validators.MutuallyExclusive(),  # Only one option is allowed to be selected.
)


@Parameter(name="*")
@dataclass
class Common:
    quiet: Annotated[
        bool,
        Parameter(group=verbosity_group, help="Suppress all output except errors."),
    ] = False
    verbose: Annotated[
        bool, Parameter(group=verbosity_group, help="Print additional information.")
    ] = False


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


def set_logging_level(common: Common | None = None):
    if common and common.quiet:
        level = logging.ERROR
    elif common and common.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    setup_logging(level)


@error_handler
def project_structure_svg(
    structure: Path,
    output: Annotated[Optional[Path], Parameter(name=["-o", "--output"])] = None,
    matrix: Optional[Path] = None,
    style: Optional[Path] = None,
    annotations: Optional[Path] = None,
    dssp: Optional[Path] = None,
    *,
    common: Common | None = None,
) -> int:
    """
    Generate a 2D projection of a protein structure.

    This function processes a protein structure file, applies transformations, styles, and annotations, and generates a 2D SVG visualization.

    Args:
        structure: Path to the structure file (PDB or similar).
            Supported formats include PDB (.pdb) and mmCIF (.cif, .mmcif).
            The file must exist and be in a valid format.
        output: Path to save the SVG output.
            If not provided, the SVG is printed to stdout.
            The directory will be created if it doesn't exist.
        matrix: Path to a custom transformation matrix.
            If not provided, a default inertia transformation is used.
            The matrix should be a NumPy (.npy) file containing (a) A 4x3 matrix where the first 3 rows are the rotation matrix (3x3) and the last row is the translation vector (1x3) (b) Alternatively, a 3x3 rotation matrix (translation will be set to zero) (c) The matrix can also be transposed (3x4) and will be automatically corrected
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
    set_logging_level(common)

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
def align_structure_rotation(
    structure_file: Path,
    matrix_out_path: Annotated[
        Path, Parameter(name=["matrix-out-path", "-m", "--matrix"])
    ] = Path("alignment_matrix.npy"),
    info_out_path: Annotated[
        Optional[Path], Parameter(name=["info-out-path", "-i", "--info"])
    ] = None,
    foldseek_path: Annotated[
        str, Parameter(name=["foldseek-path", "-f", "--foldseek"])
    ] = "foldseek",
    database_path: Annotated[
        Optional[Path], Parameter(name=["database-path", "-d", "--database"])
    ] = None,
    database_file_name: Annotated[
        str, Parameter(name=["database-file-name", "-n"])
    ] = "alignments.h5",
    foldseek_db_path: Annotated[
        Optional[Path], Parameter(name=["foldseek-db-path", "-b", "--foldseek-db"])
    ] = None,
    min_probability: Annotated[
        float,
        Parameter(
            validator=validators.Number(gt=0, lt=1.0),
            name=["min-probability", "-p", "--min-probability"],
        ),
    ] = 0.5,
    download_db: bool = False,
    *,
    common: Common | None = None,
) -> int:
    """
    Align a protein structure to known superfamilies.

    This command finds the best matching protein superfamily for the given structure
    and returns alignment information including rotation matrices.

    Args:
        structure_file: Path to the input structure file (PDB or similar).
            Supported formats include PDB (.pdb) and mmCIF (.cif, .mmcif).
            The file must exist and be in a valid format.

        matrix_out_path: Path to save the transformation matrix.
            Identified alignment matrix will be saved in this file.

        info_out_path: Path to save the alignment information.
            If not provided, the results are printed to stdout.
            The directory will be created if it doesn't exist.

        foldseek_path: Path to the FoldSeek executable.
            Defaults to "foldseek" which assumes it's in the system PATH.
            Set this if FoldSeek is installed in a non-standard location.

        foldseek_db_path: Path to a custom foldseek database.
            If not provided, the default database will be used.
            If the database doesn't exist, it will be downloaded unless download_db is False.

        database_path: Path to the directory containing the custom alignment database.
            If not provided, the default database will be used.
            If the database doesn't exist, it will be downloaded unless download_db is False.

        database_file_name: Name of the alignment database file.
            Defaults to "alignments.h5".

        min_probability: Minimum probability threshold for alignments.
            Alignments with probability below this threshold will be rejected.
            Value should be larger than 0.0 and smaller than 1.0, with higher values being more stringent.

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
    set_logging_level(common)

    if not Path(foldseek_path).exists() and not shutil.which(foldseek_path):
        raise RuntimeError(f"FoldSeek executable not found: {foldseek_path}")

    try:
        # Validate structure file using shared validation
        validate_structure_file(structure_file)

        # Database handling
        db_path = ensure_database_available(database_path, download_db)
        db_file_path = db_path / database_file_name

        logger.debug(f"Using alignment database at: {db_file_path}")

        foldseek_db_path = foldseek_db_path or db_path / "foldseek" / "db"

        logger.debug(f"Using foldseek database: {foldseek_db_path}")

        # Initialize database
        alignment_db = AlignmentDatabase(db_file_path)

        # Logging configuration matches project_structure_svg pattern
        logger.info(f"Aligning structure: {structure_file.name}")
        logger.debug(f"Using foldseek database: {foldseek_db_path}")
        logger.debug(f"FoldSeek path: {foldseek_path}")

        # Alignment process
        alignment_result = align_structure_database(
            structure_file=structure_file,
            foldseek_db_path=foldseek_db_path,
            foldseek_command=foldseek_path,
            min_probability=min_probability,
        )

        # Matrix combination
        final_matrix, db_entry = get_aligned_rotation_database(
            alignment_result, alignment_db
        )

        save_alignment_matrix(final_matrix, Path(matrix_out_path))
        logger.info(f"Saved rotation matrix to {matrix_out_path}")

        save_alignment_results(
            result=alignment_result,
            db_entry=db_entry,
            output_path=info_out_path,
            structure_file=structure_file,
        )

        logger.info("Alignment completed successfully")
        return 0

    except FlatProtError as e:
        logger.error(e.message)
        return 1
    except NoSignificantAlignmentError as e:
        logger.warning(str(e))
        logger.info("Try lowering the --min-probability threshold")
        return 1
    except DatabaseEntryNotFoundError as e:
        logger.error(str(e))
        logger.info("This could indicate database corruption. Try --download-db")
        return 1
    except (FileNotFoundError, InvalidStructureError) as e:
        logger.error(f"Input error: {str(e)}")
        return 1
    except OutputFileError as e:
        logger.error(str(e))
        return 1
    except subprocess.SubprocessError as e:
        logger.error(f"FoldSeek execution failed: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug("Stack trace:", exc_info=True)
        return 1
