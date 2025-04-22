# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""
Create the FlatProt alignment database from representative domain structures.

This script:
1. Reads representative domain structures passed via Snakemake input
2. Calculates inertia transformation matrices for each structure
3. Creates an HDF5 alignment database compatible with AlignmentDatabase
4. Generates database metadata

Usage in Snakemake:
    rule create_alignment_database:
        input:
            representatives = [... list of representative cif files ...],
            superfamilies = "path/to/superfamilies.tsv" # Optional for metadata
        output:
            database = "path/to/alignments.h5",
            database_info = "path/to/database_info.json"
        script:
            "db_create.py"
"""

import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import polars as pl
import gemmi
from snakemake.script import snakemake

# Import required FlatProt modules
from flatprot.transformation import TransformationMatrix
from flatprot.transformation.inertia_transformation import (
    calculate_inertia_transformation_matrix,
)
from flatprot.alignment.db import AlignmentDBEntry, AlignmentDatabase
from flatprot.core.types import ResidueType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=snakemake.log[0],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_inertia_matrix_from_pdb(pdb_file: Path) -> Optional[TransformationMatrix]:
    """
    Calculate the inertia transformation matrix from a structure file (PDB/CIF).

    Args:
        pdb_file: Path to the structure file (expecting CIF from this pipeline).

    Returns:
        Optional[TransformationMatrix]: The inertia transformation matrix or None if calculation fails.
    """
    try:
        # Read the structure with gemmi (assuming CIF but Detect should work)
        structure_gemmi = gemmi.read_structure(
            str(pdb_file), format=gemmi.CoorFormat.Detect
        )

        # Extract C-alpha coordinates and corresponding residue types
        ca_coords = []
        residue_types = []  # Keep track for potential future weighted calculations

        # Assume single model, single chain for representative domains
        if not structure_gemmi:
            logger.warning(f"Gemmi could not read structure: {pdb_file}")
            return None
        if len(structure_gemmi) == 0:
            logger.warning(f"No models found in structure: {pdb_file}")
            return None
        model = structure_gemmi[0]
        if len(model) == 0:
            logger.warning(f"No chains found in model 0 of structure: {pdb_file}")
            return None
        chain = model[0]

        for residue in chain:
            # Try to map PDB residue name to our Residue enum
            try:
                res_name = residue.name.strip()
                res_type = ResidueType[res_name]
            except (KeyError, ValueError):
                # If not a standard amino acid, assign default type or handle as needed
                res_type = None

            ca_atom = residue.find_atom("CA", "\0")  # Find C-alpha atom
            if ca_atom:
                pos = ca_atom.pos
                ca_coords.append([pos.x, pos.y, pos.z])
                residue_types.append(res_type)
            # else: log warning? Missing CA atoms might indicate issues.
            # logger.warning(f"Residue {residue.seqid.num} in {pdb_file} missing CA atom.")

        if not ca_coords:
            logger.warning(f"No C-alpha atoms found in {pdb_file}")
            return None

        # Convert to numpy array
        ca_coords_array = np.array(ca_coords)

        # Generate weights array (currently uniform weights)
        # TODO: Consider using actual amino acid masses if needed later
        weights = np.ones(len(ca_coords))

        # Calculate inertia transformation with weights
        transformation = calculate_inertia_transformation_matrix(
            ca_coords_array, weights
        )

        return transformation

    except Exception as e:
        # Catch potential gemmi errors or other issues
        logger.error(
            f"Failed to calculate inertia transformation for {pdb_file}: {e}",
            exc_info=True,
        )
        return None


def load_superfamily_metadata(
    superfamilies_file: Optional[Path],
) -> Dict[str, Dict[str, int]]:
    """
    Load superfamily metadata (domain/PDB counts) from the superfamilies TSV file.

    Args:
        superfamilies_file: Optional path to the superfamilies TSV file.

    Returns:
        A dictionary mapping sf_id (as string) to its metadata, or empty dict if file is None or invalid.
    """
    sf_metadata = {}
    if superfamilies_file and superfamilies_file.exists():
        try:
            df = pl.read_csv(superfamilies_file, separator="\t")
            # Ensure sf_id is treated as string for consistency with file names
            df = df.with_columns(pl.col("sf_id").cast(pl.Utf8))

            # Group by superfamily ID
            grouped = df.group_by("sf_id").agg(
                [
                    pl.len().alias(
                        "domain_count"
                    ),  # Count all domain entries for the SF
                    pl.col("pdb_id")
                    .n_unique()
                    .alias("pdb_count"),  # Count unique PDB IDs within the SF
                ]
            )

            for row in grouped.to_dicts():
                sf_metadata[row["sf_id"]] = {
                    "domain_count": row["domain_count"],
                    "pdb_count": row["pdb_count"],
                }

            logger.info(
                f"Loaded metadata for {len(sf_metadata)} superfamilies from {superfamilies_file}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load or parse superfamily metadata from {superfamilies_file}: {e}"
            )
            sf_metadata = {}  # Reset on error
    else:
        logger.info(
            "Superfamilies file not provided or not found. Proceeding without superfamily metadata."
        )

    return sf_metadata


def create_alignment_database(
    representative_domain_files: List[Path],
    output_database: Path,
    database_info_file: Path,
    superfamilies_file: Optional[Path] = None,
) -> None:
    """
    Create the FlatProt alignment database with inertia transformations.

    Args:
        representative_domain_files: List of Paths to representative domain CIF files.
        output_database: Path to the output HDF5 database file.
        database_info_file: Path to the output database info JSON file.
        superfamilies_file: Optional path to superfamilies info TSV file for metadata.

    Returns:
        None
    """
    successful_entries = 0
    failed_entries = 0
    total_representatives = len(representative_domain_files)

    # Load superfamily metadata if file is provided
    sf_metadata = load_superfamily_metadata(superfamilies_file)

    if not representative_domain_files:
        logger.warning(
            "No representative domain files provided. Creating empty database."
        )
        # Ensure output directory exists
        output_database.parent.mkdir(parents=True, exist_ok=True)
        # Create empty HDF5 file using the AlignmentDatabase context manager
        try:
            with AlignmentDatabase(output_database) as db:
                pass  # Context manager handles creation
            logger.info(f"Created empty alignment database at {output_database}")
        except Exception as e:
            logger.error(f"Failed to create empty alignment database: {e}")
            sys.exit(1)
    else:
        logger.info(
            f"Creating alignment database from {total_representatives} representative domain files."
        )
        # Create the database using the AlignmentDatabase class
        try:
            # Delete existing database if it exists
            if output_database.exists():
                logger.info(f"Removing existing database file: {output_database}")
                output_database.unlink()
            else:
                # Ensure parent directory exists if file doesn't exist
                output_database.parent.mkdir(parents=True, exist_ok=True)

            # Create database and populate it with entries
            with AlignmentDatabase(output_database) as db:
                for domain_file in representative_domain_files:
                    # SF ID is the filename without extension (e.g., "12345.cif" -> "12345")
                    sf_id = domain_file.stem
                    entry_id = sf_id  # Use sf_id as the unique entry ID in the DB
                    structure_name = sf_id  # Use sf_id as the structure name

                    # Calculate inertia transformation matrix
                    transformation = calculate_inertia_matrix_from_pdb(domain_file)

                    if transformation is None:
                        logger.warning(
                            f"Skipping SF ID {sf_id} (from file {domain_file.name}) due to transformation calculation failure."
                        )
                        failed_entries += 1
                        continue

                    # Get metadata for this specific SF ID, if available
                    specific_sf_meta = sf_metadata.get(
                        sf_id, {}
                    )  # Returns empty dict if not found

                    # Create and add entry
                    entry = AlignmentDBEntry(
                        rotation_matrix=transformation,
                        entry_id=entry_id,
                        structure_name=structure_name,
                        metadata=specific_sf_meta,  # Add the loaded metadata here
                    )

                    try:
                        db.add_entry(entry)
                        successful_entries += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to add entry for SF ID {sf_id} to database: {e}",
                            exc_info=True,
                        )
                        failed_entries += 1

            logger.info(f"Successfully created alignment database at {output_database}")
            logger.info(
                f"Added {successful_entries} entries, {failed_entries} failed out of {total_representatives} representatives."
            )
            if failed_entries > 0:
                logger.warning(
                    f"{failed_entries} representative domains could not be processed."
                )

        except Exception as e:
            logger.error(
                f"Failed to create or populate alignment database: {e}", exc_info=True
            )
            sys.exit(1)

    # Create database info file (always, even if DB is empty)
    database_info: Dict[str, Any] = {
        "creation_date": datetime.now().isoformat(),
        "entry_count": successful_entries,  # Number of entries successfully added
        "total_representatives_processed": total_representatives,  # Total number of files passed as input
        "failed_representatives": failed_entries,
        "superfamily_metadata_included": bool(sf_metadata),
        "superfamily_count_from_metadata": len(sf_metadata) if sf_metadata else 0,
        "format_version": snakemake.params.db_version,
        "transformation_type": "inertia",
    }

    try:
        database_info_file.parent.mkdir(parents=True, exist_ok=True)
        with open(database_info_file, "w") as f:
            json.dump(database_info, f, indent=2)
        logger.info(f"Created database info file at {database_info_file}")
    except Exception as e:
        logger.error(f"Failed to write database info file {database_info_file}: {e}")
        # Don't necessarily exit, DB creation might have succeeded


def main() -> None:
    """
    Main entry point for the Snakemake script.

    Reads input files and output paths from the snakemake object
    and calls the database creation function.
    """
    # --- Inputs ---
    # Get the list of representative domain CIF files from snakemake input
    representative_files_str: List[str] = snakemake.input.representatives
    representative_files: List[Path] = [Path(f) for f in representative_files_str]
    logger.info(
        f"Received {len(representative_files)} representative file paths from Snakemake."
    )

    # Get the optional superfamilies file path
    superfamilies_file_str: Optional[str] = getattr(
        snakemake.input, "superfamilies", None
    )
    superfamilies_file: Optional[Path] = (
        Path(superfamilies_file_str) if superfamilies_file_str else None
    )
    if superfamilies_file:
        logger.info(f"Received superfamilies metadata file path: {superfamilies_file}")
    else:
        logger.info("No superfamilies metadata file provided in input.")

    # --- Outputs ---
    output_database = Path(snakemake.output.database)
    database_info_file = Path(snakemake.output.database_info)
    logger.info(f"Output database path: {output_database}")
    logger.info(f"Output database info path: {database_info_file}")

    # --- Create the alignment database ---
    create_alignment_database(
        representative_domain_files=representative_files,
        output_database=output_database,
        database_info_file=database_info_file,
        superfamilies_file=superfamilies_file,
    )


if __name__ == "__main__":
    # Check if running under Snakemake
    if "snakemake" not in globals():
        print("Error: This script must be run via Snakemake.")
        # You could potentially add mock Snakemake object here for testing
        # import sys
        # sys.exit(1)
        # Mock snakemake for testing purposes:
        from unittest.mock import Mock

        snakemake = Mock()
        snakemake.input = Mock(
            representatives=["test_repr_1.cif", "test_repr_2.cif"],
            superfamilies="test_superfamilies.tsv",
        )
        snakemake.output = Mock(
            database="test_output.h5", database_info="test_output.json"
        )
        snakemake.log = ["test_db_create.log"]
        # Create dummy files for mock testing if needed
        Path("test_repr_1.cif").touch()
        Path("test_repr_2.cif").touch()
        Path("test_superfamilies.tsv").touch()
        print("Running with mock Snakemake object for testing...")

    main()
