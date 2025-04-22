# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Optional, Annotated, Literal
import shutil
import subprocess

from cyclopts import Parameter, validators

from flatprot.core.errors import FlatProtError
from flatprot.cli.errors import error_handler

from flatprot.io import validate_structure_file

from flatprot.core import logger
from flatprot.utils.database import ensure_database_available

from flatprot.alignment import align_structure_database, get_aligned_rotation_database
from flatprot.alignment import AlignmentDatabase
from flatprot.alignment import (
    NoSignificantAlignmentError,
    DatabaseEntryNotFoundError,
)

from flatprot.io import OutputFileError, InvalidStructureError
from flatprot.utils.alignment import save_alignment_results, save_alignment_matrix
from .utils import set_logging_level, CommonParameters


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
    target_db_id: Annotated[Optional[str], Parameter(name=["--target-db-id"])] = None,
    alignment_mode: Annotated[
        Literal["family-identity", "family-inertia"],
        Parameter(name=["--alignment-mode", "-a"]),
    ] = "family-identity",
    *,
    common: CommonParameters | None = None,
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

        target_db_id: Align specifically to this database entry ID.
            If provided, alignment will be forced against this specific target ID
            from the Foldseek database, bypassing the probability-based selection.

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
            target_db_id=target_db_id,
        )

        combined_matrix, db_entry = get_aligned_rotation_database(
            alignment_result, alignment_db
        )

        # Matrix combination
        if alignment_mode == "family-identity":
            final_matrix = alignment_result.rotation_matrix
        elif alignment_mode == "family-inertia":
            final_matrix = combined_matrix

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

    except NoSignificantAlignmentError as e:
        logger.error(str(e))
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
    except FlatProtError as e:
        logger.error(e.message)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug("Stack trace:", exc_info=True)
        return 1
