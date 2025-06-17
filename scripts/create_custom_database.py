#!/usr/bin/env python3
"""
Create a custom FlatProt alignment database from a folder of protein structures.

This script allows users to create their own alignment databases from a collection
of protein structure files (PDB/CIF format). It supports optional structure rotation
before calculating inertia transformation matrices.

Usage:
    python create_custom_database.py <input_folder> <output_database> [options]

Example:
    python create_custom_database.py ./my_structures ./my_database.h5 --rotate-method inertia
"""

import argparse
import json
import logging
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import gemmi

from flatprot.transformation import TransformationMatrix
from flatprot.transformation.inertia_transformation import (
    calculate_inertia_transformation_matrix,
)
from flatprot.alignment.db import AlignmentDBEntry, AlignmentDatabase
from flatprot.core.types import ResidueType


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    if log_file:
        logging.basicConfig(level=level, filename=log_file, format=format_str)
    else:
        logging.basicConfig(level=level, format=format_str)


def find_structure_files(input_folder: Path) -> List[Path]:
    """
    Find all structure files (PDB/CIF) in the input folder.

    Args:
        input_folder: Path to folder containing structure files

    Returns:
        List of paths to structure files
    """
    extensions = [".pdb", ".cif", ".ent", ".pdb.gz", ".cif.gz"]
    structure_files = []

    for ext in extensions:
        structure_files.extend(input_folder.glob(f"*{ext}"))
        structure_files.extend(input_folder.glob(f"**/*{ext}"))

    return sorted(structure_files)


def extract_ca_coordinates(
    structure_file: Path,
) -> Tuple[Optional[np.ndarray], Optional[List[ResidueType]]]:
    """
    Extract C-alpha coordinates and residue types from a structure file.

    Args:
        structure_file: Path to structure file

    Returns:
        Tuple of (coordinates array, residue types list) or (None, None) if failed
    """
    try:
        structure = gemmi.read_structure(str(structure_file))

        if not structure or len(structure) == 0:
            logging.warning(f"No models found in {structure_file}")
            return None, None

        model = structure[0]
        if len(model) == 0:
            logging.warning(f"No chains found in {structure_file}")
            return None, None

        ca_coords = []
        residue_types = []

        # Process all chains
        for chain in model:
            for residue in chain:
                # Try to map residue name to ResidueType
                try:
                    res_name = residue.name.strip()
                    res_type = ResidueType[res_name]
                except (KeyError, ValueError):
                    res_type = None

                ca_atom = residue.find_atom("CA", "\0")
                if ca_atom:
                    pos = ca_atom.pos
                    ca_coords.append([pos.x, pos.y, pos.z])
                    residue_types.append(res_type)

        if not ca_coords:
            logging.warning(f"No C-alpha atoms found in {structure_file}")
            return None, None

        return np.array(ca_coords), residue_types

    except Exception as e:
        logging.error(f"Failed to extract coordinates from {structure_file}: {e}")
        return None, None


def apply_rotation(
    coords: np.ndarray,
    rotation_method: str,
    rotation_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply rotation to coordinates before calculating inertia transformation.

    Note: The preferred workflow is to manually rotate structures (e.g., in PyMOL)
    and save them in the desired orientation, then use rotation_method='none'.
    This ensures Foldseek will align correctly to your manually oriented structures.

    Args:
        coords: C-alpha coordinates
        rotation_method: Method for rotation ('none', 'custom', 'random')
        rotation_params: Additional parameters for rotation

    Returns:
        Rotated coordinates
    """
    if rotation_method == "none":
        # Use structures as-is (recommended for manually rotated structures)
        return coords

    elif rotation_method == "random":
        # Apply random rotation (mainly for testing)
        from scipy.spatial.transform import Rotation

        random_rotation = Rotation.random().as_matrix()
        # Center coordinates first
        centered = coords - coords.mean(axis=0)
        rotated = centered @ random_rotation.T
        return rotated + coords.mean(axis=0)

    elif rotation_method == "custom":
        # Apply custom rotation matrix if provided
        if rotation_params and "rotation_matrix" in rotation_params:
            rotation_matrix = np.array(rotation_params["rotation_matrix"])
            centered = coords - coords.mean(axis=0)
            rotated = centered @ rotation_matrix.T
            return rotated + coords.mean(axis=0)
        else:
            logging.warning("Custom rotation requested but no rotation_matrix provided")
            return coords

    else:
        logging.warning(f"Unknown rotation method: {rotation_method}")
        return coords


def calculate_transformation_matrix(
    structure_file: Path,
    rotation_method: str = "none",
    rotation_params: Optional[Dict[str, Any]] = None,
) -> Optional[TransformationMatrix]:
    """
    Calculate inertia transformation matrix for a structure file.

    Args:
        structure_file: Path to structure file
        rotation_method: Method for pre-rotation
        rotation_params: Parameters for rotation

    Returns:
        TransformationMatrix or None if calculation failed
    """
    coords, residue_types = extract_ca_coordinates(structure_file)
    if coords is None:
        return None

    # Apply rotation if requested
    if rotation_method != "none":
        coords = apply_rotation(coords, rotation_method, rotation_params)

    # Calculate weights (uniform for now)
    weights = np.ones(len(coords))

    try:
        transformation = calculate_inertia_transformation_matrix(coords, weights)
        return transformation
    except Exception as e:
        logging.error(f"Failed to calculate transformation for {structure_file}: {e}")
        return None


def create_foldseek_database(
    structure_files: List[Path],
    foldseek_db_path: Path,
    foldseek_executable: str = "foldseek",
) -> bool:
    """
    Create a Foldseek database from structure files.

    Args:
        structure_files: List of structure file paths
        foldseek_db_path: Path to output Foldseek database (without extension)
        foldseek_executable: Path to Foldseek executable

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if foldseek is available
        if not shutil.which(foldseek_executable):
            logging.error(f"Foldseek executable not found: {foldseek_executable}")
            return False

        # Create temporary directory with all structure files
        import tempfile

        with tempfile.TemporaryDirectory(prefix="foldseek_db_") as temp_dir:
            temp_path = Path(temp_dir)
            structures_dir = temp_path / "structures"
            structures_dir.mkdir()

            # Copy structure files to temporary directory
            for structure_file in structure_files:
                dest_file = structures_dir / structure_file.name
                shutil.copy2(structure_file, dest_file)

            # Create the output directory
            foldseek_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Run foldseek createdb
            cmd = [
                foldseek_executable,
                "createdb",
                str(structures_dir),
                str(foldseek_db_path),
            ]

            logging.info(f"Creating Foldseek database: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.returncode == 0:
                logging.info("Foldseek database created successfully")
                return True
            else:
                logging.error(f"Foldseek database creation failed: {result.stderr}")
                return False

    except subprocess.CalledProcessError as e:
        logging.error(f"Foldseek command failed: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Failed to create Foldseek database: {e}")
        return False


def create_database_info(db_dir: Path, creation_stats: Dict[str, Any]) -> None:
    """
    Create database_info.json file with database metadata.

    Args:
        db_dir: Database directory path
        creation_stats: Statistics from database creation
    """
    info = {
        "database_type": "custom_alignment",
        "version": "1.0.0",
        "creation_date": datetime.now().isoformat(),
        "description": "Custom FlatProt alignment database",
        "statistics": creation_stats,
        "files": {
            "alignment_database": "alignments.h5",
            "foldseek_database": "foldseek/db",
        },
        "requirements": ["FlatProt >= 1.0.0", "Foldseek"],
    }

    info_file = db_dir / "database_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    logging.info(f"Created database info file: {info_file}")


def create_custom_database(
    input_folder: Path,
    output_database_dir: Path,
    rotation_method: str = "none",
    rotation_params: Optional[Dict[str, Any]] = None,
    name_pattern: str = "filename",
    foldseek_executable: str = "foldseek",
) -> Dict[str, Any]:
    """
    Create a custom alignment database from structure files.

    Args:
        input_folder: Path to folder containing structure files
        output_database_dir: Path to output database directory
        rotation_method: Method for structure rotation
        rotation_params: Parameters for rotation
        name_pattern: How to generate structure names ('filename', 'parent_folder')
        foldseek_executable: Path to Foldseek executable

    Returns:
        Dictionary with creation statistics
    """
    # Find structure files
    structure_files = find_structure_files(input_folder)
    logging.info(f"Found {len(structure_files)} structure files in {input_folder}")

    if not structure_files:
        logging.error("No structure files found!")
        return {"error": "No structure files found"}

    # Create output directory
    output_database_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    alignment_db_file = output_database_dir / "alignments.h5"
    foldseek_dir = output_database_dir / "foldseek"
    foldseek_db_path = foldseek_dir / "db"

    # Remove existing files
    if alignment_db_file.exists():
        logging.info(f"Removing existing alignment database: {alignment_db_file}")
        alignment_db_file.unlink()

    if foldseek_dir.exists():
        logging.info(f"Removing existing Foldseek database directory: {foldseek_dir}")
        shutil.rmtree(foldseek_dir)

    successful_entries = 0
    failed_entries = 0

    # Create alignment database (HDF5)
    try:
        with AlignmentDatabase(alignment_db_file) as db:
            for i, structure_file in enumerate(structure_files):
                logging.info(
                    f"Processing {i+1}/{len(structure_files)}: {structure_file.name}"
                )

                # Generate entry ID and structure name
                if name_pattern == "filename":
                    entry_id = structure_file.stem
                    structure_name = structure_file.stem
                elif name_pattern == "parent_folder":
                    entry_id = f"{structure_file.parent.name}_{structure_file.stem}"
                    structure_name = (
                        f"{structure_file.parent.name}_{structure_file.stem}"
                    )
                else:
                    entry_id = structure_file.stem
                    structure_name = structure_file.stem

                # Calculate transformation matrix
                transformation = calculate_transformation_matrix(
                    structure_file, rotation_method, rotation_params
                )

                if transformation is None:
                    logging.warning(
                        f"Skipping {structure_file.name} - transformation failed"
                    )
                    failed_entries += 1
                    continue

                # Create database entry
                entry = AlignmentDBEntry(
                    rotation_matrix=transformation,
                    entry_id=entry_id,
                    structure_name=structure_name,
                    metadata={
                        "source_file": str(structure_file),
                        "rotation_method": rotation_method,
                        "file_size": structure_file.stat().st_size,
                    },
                )

                try:
                    db.add_entry(entry)
                    successful_entries += 1
                except Exception as e:
                    logging.error(f"Failed to add entry {entry_id}: {e}")
                    failed_entries += 1

        logging.info(
            f"Alignment database creation completed: {successful_entries} success, {failed_entries} failed"
        )

    except Exception as e:
        logging.error(f"Failed to create alignment database: {e}")
        return {"error": str(e)}

    # Create Foldseek database
    logging.info("Creating Foldseek database...")
    foldseek_success = create_foldseek_database(
        [
            f for f in structure_files if f.stat().st_size > 0
        ],  # Only valid structure files
        foldseek_db_path,
        foldseek_executable,
    )

    if not foldseek_success:
        logging.error("Failed to create Foldseek database")
        return {"error": "Failed to create Foldseek database"}

    # Create database info file
    creation_stats = {
        "creation_date": datetime.now().isoformat(),
        "input_folder": str(input_folder),
        "total_files": len(structure_files),
        "successful_entries": successful_entries,
        "failed_entries": failed_entries,
        "rotation_method": rotation_method,
        "transformation_type": "inertia",
    }

    create_database_info(output_database_dir, creation_stats)

    logging.info("Custom database creation completed successfully!")

    # Return statistics
    return creation_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create custom FlatProt alignment database from structure files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic database creation (recommended - structures should be pre-rotated)
  python create_custom_database.py ./structures ./my_database_dir

  # With random rotation (mainly for testing)
  python create_custom_database.py ./structures ./my_database_dir --rotate-method random

  # Include parent folder in names
  python create_custom_database.py ./structures ./my_database_dir --name-pattern parent_folder

Recommended Workflow:
  1. Manually rotate structures in PyMOL/ChimeraX to desired orientation
  2. Save rotated structures as PDB/CIF files
  3. Use default --rotate-method none to preserve manual orientation
        """,
    )

    parser.add_argument(
        "input_folder", type=Path, help="Folder containing protein structure files"
    )
    parser.add_argument(
        "output_database_dir", type=Path, help="Output database directory path"
    )

    parser.add_argument(
        "--rotate-method",
        choices=["none", "random", "custom"],
        default="none",
        help="Method for rotating structures before transformation (default: none, recommended for manually pre-rotated structures)",
    )

    parser.add_argument(
        "--name-pattern",
        choices=["filename", "parent_folder"],
        default="filename",
        help="How to generate structure names (default: filename)",
    )

    parser.add_argument(
        "--foldseek-executable",
        type=str,
        default="foldseek",
        help="Path to Foldseek executable (default: foldseek)",
    )

    parser.add_argument("--log-file", type=Path, help="Log file path (default: stdout)")

    parser.add_argument(
        "--info-file", type=Path, help="Output JSON file with database creation info"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file, args.verbose)

    # Validate inputs
    if not args.input_folder.exists():
        logging.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    if not args.input_folder.is_dir():
        logging.error(f"Input path is not a directory: {args.input_folder}")
        sys.exit(1)

    # Check Foldseek availability
    if not shutil.which(args.foldseek_executable):
        logging.error(f"Foldseek executable not found: {args.foldseek_executable}")
        logging.error(
            "Please install Foldseek or provide the correct path with --foldseek-executable"
        )
        sys.exit(1)

    # Create database
    logging.info(f"Creating custom database from {args.input_folder}")
    logging.info(f"Output database directory: {args.output_database_dir}")
    logging.info(f"Rotation method: {args.rotate_method}")
    logging.info(f"Foldseek executable: {args.foldseek_executable}")

    result = create_custom_database(
        input_folder=args.input_folder,
        output_database_dir=args.output_database_dir,
        rotation_method=args.rotate_method,
        name_pattern=args.name_pattern,
        foldseek_executable=args.foldseek_executable,
    )

    if "error" in result:
        logging.error(f"Database creation failed: {result['error']}")
        sys.exit(1)

    # Save info file if requested
    if args.info_file:
        try:
            with open(args.info_file, "w") as f:
                json.dump(result, f, indent=2)
            logging.info(f"Database info saved to {args.info_file}")
        except Exception as e:
            logging.error(f"Failed to save info file: {e}")

    logging.info("Database creation completed successfully!")
    print(f"Created database with {result['successful_entries']} entries")
    if result["failed_entries"] > 0:
        print(f"Warning: {result['failed_entries']} files failed to process")

    print(f"\nDatabase created at: {args.output_database_dir}")
    print(f"  - Alignment database: {args.output_database_dir}/alignments.h5")
    print(f"  - Foldseek database: {args.output_database_dir}/foldseek/db")
    print(f"  - Database info: {args.output_database_dir}/database_info.json")

    print("\nTo use this database with FlatProt:")
    print(f"  flatprot align structure.cif --database {args.output_database_dir}")


if __name__ == "__main__":
    main()
