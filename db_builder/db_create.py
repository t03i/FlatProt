# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""
Create the FlatProt alignment database from representative domain structures.

This script:
1. Reads representative domain structures
2. Calculates inertia transformation matrices for each structure
3. Creates an HDF5 alignment database compatible with AlignmentDatabase
4. Generates database metadata

Usage in Snakemake:
    rule create_alignment_database:
        input:
            directory(REPRESENTATIVE_DOMAINS)
        output:
            database = f"{ALIGNMENT_DB}",
            database_info = f"{DATABASE_INFO}"
        script:
            "create_alignment_db.py"
"""

import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import gemmi
from snakemake.script import snakemake

# Import required FlatProt modules
from flatprot.transformation import TransformationMatrix
from flatprot.transformation.inertia import calculate_inertia_transformation
from flatprot.alignment.db import AlignmentDBEntry, AlignmentDatabase
from flatprot.core.residue import Residue
from flatprot.transformation.inertia import InertiaTransformerParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=snakemake.log[0],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_inertia_matrix_from_pdb(pdb_file: Path) -> Optional[TransformationMatrix]:
    """
    Calculate the inertia transformation matrix from a PDB file.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        Optional[TransformationMatrix]: The inertia transformation matrix or None if calculation fails
    """
    try:
        # Get default residue weights
        default_params = InertiaTransformerParameters.default()
        residue_weights = default_params.residue_weights

        # Read the structure with gemmi
        structure_gemmi = gemmi.read_structure(str(pdb_file))

        # Extract C-alpha coordinates and corresponding residue types
        ca_coords = []
        residue_types = []

        for model in structure_gemmi:
            for chain in model:
                for residue in chain:
                    # Try to map PDB residue name to our Residue enum
                    try:
                        res_name = residue.name.strip()
                        res_type = Residue[res_name]
                    except (KeyError, ValueError):
                        # If not a standard amino acid, use default weight
                        res_type = None

                    for atom in residue:
                        if atom.name == "CA":  # Only get C-alpha atoms
                            pos = atom.pos
                            ca_coords.append([pos.x, pos.y, pos.z])
                            residue_types.append(res_type)

        # Convert to numpy array
        ca_coords_array = np.array(ca_coords)

        # Generate weights array based on residue types
        weights = np.ones(len(ca_coords))  # Default weight is 1.0

        for i, res_type in enumerate(residue_types):
            if res_type is not None and res_type in residue_weights:
                weights[i] = residue_weights[res_type]

        # Calculate inertia transformation with weights
        transformation = calculate_inertia_transformation(ca_coords_array, weights)

        return transformation

    except Exception as e:
        logger.error(f"Failed to calculate inertia transformation for {pdb_file}: {e}")
        return None


def create_alignment_database(
    domain_files: list[Path],
    output_database: Path,
    database_info_file: Path,
    superfamilies_file: Optional[Path] = None,
) -> None:
    """
    Create the FlatProt alignment database with inertia transformations.

    Args:
        representative_domains_dir: Directory containing representative domain PDB files
        output_database: Path to the output HDF5 database file
        database_info_file: Path to the output database info JSON file
        superfamilies_file: Optional path to superfamilies info TSV file

    Returns:
        None
    """
    if not domain_files:
        logger.error("No domain files supplied")
        sys.exit(1)

    logger.info(f"Creating alignment database from {len(domain_files)} domain files")

    # Load superfamily information if available
    sf_metadata = {}
    if superfamilies_file and superfamilies_file.exists():
        try:
            df = pl.read_csv(superfamilies_file, separator="\t")
            # Group by superfamily ID
            grouped = df.group_by("sf_id").agg(
                [
                    pl.count().alias("domain_count"),
                    pl.col("pdb_id").unique().count().alias("pdb_count"),
                ]
            )

            for row in grouped.to_dicts():
                sf_metadata[row["sf_id"]] = {
                    "domain_count": row["domain_count"],
                    "pdb_count": row["pdb_count"],
                }

            logger.info(f"Loaded metadata for {len(sf_metadata)} superfamilies")
        except Exception as e:
            logger.warning(f"Failed to load superfamily metadata: {e}")

    # Create the database using the AlignmentDatabase class
    try:
        # Delete existing database if it exists
        if output_database.exists():
            output_database.unlink()

        # Create database and populate it with entries
        with AlignmentDatabase(output_database) as db:
            successful_entries = 0
            failed_entries = 0

            for domain_file in domain_files:
                sf_id = domain_file.stem  # SF ID is the filename without extension

                # Create entry ID and structure name
                entry_id = f"sf_{sf_id}"
                structure_name = f"superfamily_{sf_id}"

                # Calculate inertia transformation matrix
                transformation = calculate_inertia_matrix_from_pdb(domain_file)

                if transformation is None:
                    logger.warning(
                        f"Skipping {sf_id} due to transformation calculation failure"
                    )
                    failed_entries += 1
                    continue

                # Create and add entry
                entry = AlignmentDBEntry(
                    rotation_matrix=transformation,
                    entry_id=entry_id,
                    structure_name=structure_name,
                )

                # Add entry to database
                try:
                    db.add_entry(entry)
                    successful_entries += 1
                except Exception as e:
                    logger.error(f"Failed to add entry for {sf_id}: {e}")
                    failed_entries += 1

        logger.info(f"Successfully created alignment database at {output_database}")
        logger.info(
            f"Added {successful_entries} entries to database, {failed_entries} failed"
        )

    except Exception as e:
        logger.error(f"Failed to create alignment database: {e}")
        sys.exit(1)

    # Create database info file
    database_info = {
        "database_path": str(output_database),
        "creation_date": datetime.now().isoformat(),
        "entry_count": successful_entries,
        "superfamily_count": len(domain_files),
        "format_version": "1.0",
        "transformation_type": "inertia",
    }

    with open(database_info_file, "w") as f:
        json.dump(database_info, f, indent=2)

    logger.info(f"Created database info file at {database_info_file}")


def main() -> None:
    """
    Main entry point for the Snakemake script.

    Returns:
        None
    """
    # Get input/output paths from Snakemake
    representative_domain_files = list(
        map(Path, snakemake.input.representative_domains)
    )
    output_database = Path(snakemake.output.database)
    database_info_file = Path(snakemake.output.database_info)

    # Look for superfamilies file in the output directory
    superfamilies_file = Path(os.path.dirname(output_database)) / "superfamilies.tsv"
    if not superfamilies_file.exists():
        superfamilies_file = None

    # Create the alignment database
    create_alignment_database(
        domain_files=representative_domain_files,
        output_database=output_database,
        database_info_file=database_info_file,
        superfamilies_file=superfamilies_file,
    )


if __name__ == "__main__":
    main()
