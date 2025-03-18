#!/usr/bin/env python3
# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""
Script to create a FlatProt alignment database from SCOP superfamily data.

This script:
1. Downloads PDB files for SCOP superfamilies
2. Creates a FoldSeek database of representative structures
3. Builds an alignment database with rotation matrices for FlatProt

Usage:
    python create_alignment_database.py --output /path/to/output --scop_file scop-cla-latest.txt
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys
import numpy as np
import polars as pl  # Replace pandas with polars

import urllib.request
import gzip
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

# Import directly from flatprot
from flatprot.alignment import AlignmentDatabase, AlignmentDBEntry
from flatprot.transformation import TransformationMatrix
from flatprot.transformation.utils import calculate_inertia_transformation
import gemmi  # Used for domain extraction and structure manipulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create FlatProt alignment database")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the created database",
    )
    parser.add_argument(
        "--scop_file",
        type=Path,
        default=None,
        help="Path to SCOP classification file (will download latest if not provided)",
    )
    parser.add_argument(
        "--foldseek", type=str, default="foldseek", help="Path to FoldSeek executable"
    )
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=None,
        help="Temporary directory for processing (uses system temp if not provided)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if output directory exists",
    )
    parser.add_argument(
        "--keep_all",
        action="store_true",
        help="Keep all intermediate files (by default, only essential files are kept)",
    )
    return parser.parse_args()


def download_scop_file(output_path: Path) -> None:
    """Download the latest SCOP classification file."""
    scop_url = "https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt"
    logger.info(f"Downloading SCOP classification from {scop_url}")

    urllib.request.urlretrieve(scop_url, output_path)
    logger.info(f"SCOP classification downloaded to {output_path}")


def get_sf_proteins_domains_region(
    scop_file: Path,
) -> Dict[str, List[Tuple[str, str, int, int]]]:
    """Extract SCOP superfamily information from 2022 SCOP classification file format using polars.

    Args:
        scop_file: Path to the SCOP classification file

    Returns:
        Dictionary mapping superfamily IDs to lists of (pdb_id, chain, start_res, end_res) tuples
    """
    logger.info(f"Parsing SCOP classification file: {scop_file}")

    # Skip header comments and read data with polars
    with open(scop_file, "r") as f:
        header_lines = 0
        for line in f:
            if line.startswith("#"):
                header_lines += 1
            else:
                break

    # Read the file with polars, skipping header lines
    df = pl.read_csv(
        scop_file,
        separator=" ",
        has_header=False,
        skip_rows=header_lines,
        new_columns=[
            "FA_DOMID",
            "FA_PDBID",
            "FA_PDBREG",
            "FA_UNIID",
            "FA_UNIREG",
            "SF_DOMID",
            "SF_PDBID",
            "SF_PDBREG",
            "SF_UNIID",
            "SF_UNIREG",
            "SCOPCLA",
        ],
    )

    # Extract superfamily IDs from the SCOPCLA column
    df = df.with_columns(
        pl.col("SCOPCLA").str.extract_all("SF=(\d+)").list.get(0).alias("sf_id")
    )

    # Process PDB regions to extract chain and residue range
    df = df.with_columns(
        [
            # Extract chain (characters before the colon)
            pl.col("SF_PDBREG").str.extract("([A-Za-z0-9]+):").alias("chain"),
            # Extract start residue (first number after colon)
            pl.col("SF_PDBREG")
            .str.extract(":(\d+)-")
            .cast(pl.Int32)
            .alias("start_res"),
            # Extract end residue (number after dash)
            pl.col("SF_PDBREG").str.extract("-(\d+)").cast(pl.Int32).alias("end_res"),
        ]
    )

    # Convert to lowercase for consistency
    df = df.with_columns(pl.col("SF_PDBID").str.to_lowercase().alias("pdb_id"))

    # Group by sf_id and collect PDB information
    grouped = df.group_by("sf_id").agg(
        [pl.struct(["pdb_id", "chain", "start_res", "end_res"]).alias("domain_info")]
    )

    # Convert to Python dictionary
    superfamilies = {}
    for row in grouped.to_dicts():
        sf_id = row["sf_id"]
        domains = []
        for domain in row["domain_info"]:
            domains.append(
                (
                    domain["pdb_id"],
                    domain["chain"],
                    domain["start_res"],
                    domain["end_res"],
                )
            )
        superfamilies[sf_id] = domains

    logger.info(f"Found {len(superfamilies)} SCOP superfamilies")
    return superfamilies


async def download_pdb_file_async(
    pdb_id: str, output_dir: Path, client: httpx.AsyncClient
) -> Optional[Path]:
    """Download a PDB file asynchronously using httpx.

    Args:
        pdb_id: The PDB ID to download
        output_dir: Directory to save the file
        client: Shared httpx AsyncClient

    Returns:
        Path to the downloaded file or None if download failed
    """
    pdb_id = pdb_id.lower()
    output_file = output_dir / f"{pdb_id}.pdb"

    if output_file.exists():
        return output_file

    # Try RCSB PDB first
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    tmp_file = output_dir / f"{pdb_id}.pdb.gz"

    try:
        # Asynchronously download the file
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()

        # Save the compressed file
        with open(tmp_file, "wb") as f:
            f.write(response.content)

        # Decompress
        with gzip.open(tmp_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                f_out.write(f_in.read())

        # Clean up
        tmp_file.unlink()
        return output_file

    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()
        logger.warning(f"Failed to download {pdb_id}: {e}")
        return None


async def batch_download_pdb_files(
    pdb_ids: List[str], output_dir: Path, max_concurrent: int = 10
) -> Dict[str, Optional[Path]]:
    """Download multiple PDB files in parallel.

    Args:
        pdb_ids: List of PDB IDs to download
        output_dir: Directory to save the files
        max_concurrent: Maximum number of concurrent downloads

    Returns:
        Dictionary mapping PDB IDs to file paths (or None if download failed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use connection pooling and limits to avoid overwhelming the server
    limits = httpx.Limits(max_connections=max_concurrent, max_keepalive_connections=5)

    results = {}

    # Process in batches to show progress and limit concurrent requests
    batch_size = 50
    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i : i + batch_size]
        logger.info(
            f"Downloading batch {i // batch_size + 1}/{(len(pdb_ids) + batch_size - 1) // batch_size}"
        )

        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            tasks = [
                download_pdb_file_async(pdb_id, output_dir, client) for pdb_id in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            # Map results back to PDB IDs
            for pdb_id, result in zip(batch, batch_results):
                results[pdb_id] = result

    return results


def download_pdb_file(pdb_id: str, output_dir: Path) -> Optional[Path]:
    """Download a PDB file from the PDB database (synchronous version).

    This function is kept for backward compatibility.
    For batch downloads, use batch_download_pdb_files instead.
    """
    pdb_id = pdb_id.lower()
    output_file = output_dir / f"{pdb_id}.pdb"

    if output_file.exists():
        return output_file

    # Try RCSB PDB first
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"

    try:
        # Download compressed file
        tmp_file = output_dir / f"{pdb_id}.pdb.gz"
        urllib.request.urlretrieve(url, tmp_file)

        # Decompress
        with gzip.open(tmp_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                f_out.write(f_in.read())

        # Clean up
        tmp_file.unlink()
        return output_file

    except Exception as e:
        logger.warning(f"Failed to download {pdb_id}: {e}")
        return None


def extract_domain(
    pdb_file: Path, chain: str, start_res: int, end_res: int, output_file: Path
) -> bool:
    """
    Extract a domain from a PDB file using gemmi.

    Args:
        pdb_file: Path to the input PDB file
        chain: Chain identifier
        start_res: Start residue number
        end_res: End residue number
        output_file: Path to save the extracted domain

    Returns:
        Boolean indicating success
    """
    try:
        # Parse PDB file with gemmi
        structure = gemmi.read_structure(str(pdb_file))

        # Create a new structure for the domain
        domain = gemmi.Structure()
        domain.name = f"{pdb_file.stem}_{chain}_{start_res}_{end_res}"

        # Create a new model with a fixed name (not using structure[0].name)
        model = gemmi.Model("1")

        # Find and copy the specified chain
        for original_chain in structure[0]:
            if original_chain.name == chain:
                new_chain = gemmi.Chain(chain)

                # Copy residues in the specified range
                for residue in original_chain:
                    seq_id = residue.seqid.num
                    if start_res <= seq_id <= end_res:
                        new_chain.add_residue(residue)

                if len(new_chain) > 0:
                    model.add_chain(new_chain)

        domain.add_model(model)

        # Save domain to PDB file
        domain.write_pdb(str(output_file))
        return True

    except Exception as e:
        logger.warning(f"Failed to extract domain from {pdb_file}: {e}")
        return False


def create_foldseek_database(
    pdb_dir: Path, output_dir: Path, foldseek_path: str
) -> Path:
    """Create a FoldSeek database from PDB files."""
    db_path = output_dir / "db"

    cmd = [foldseek_path, "createdb", str(pdb_dir), str(db_path)]

    logger.info(f"Creating FoldSeek database: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return db_path


def run_foldseek_searches(
    pdb_dir: Path, db_path: Path, output_dir: Path, tmp_dir: Path, foldseek_path: str
) -> None:
    """Run all-vs-all structure comparison using FoldSeek."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each PDB file
    for pdb_file in pdb_dir.glob("*.pdb"):
        filename = pdb_file.stem
        result_file = output_dir / filename

        cmd = [
            foldseek_path,
            "easy-search",
            str(pdb_file),
            str(db_path),
            str(result_file),
            str(tmp_dir),
            "--format-output",
            "query,target,qstart,qend,tstart,tend,tseq,prob,alntmscore,u,t,lddtfull",
        ]

        logger.info(f"Comparing {filename}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def calculate_average_scores(
    results_dir: Path, score_field: str = "prob"
) -> List[Tuple[str, float]]:
    """Calculate average alignment scores for each structure using polars."""
    averages = {}

    for result_file in results_dir.glob("*"):
        if not result_file.is_file():
            continue

        try:
            # Use polars to read the CSV file
            df = pl.read_csv(
                result_file,
                separator="\t",
                has_header=False,
                new_columns=[
                    "query",
                    "target",
                    "qstart",
                    "qend",
                    "tstart",
                    "tend",
                    "tseq",
                    "prob",
                    "alntmscore",
                    "u",
                    "t",
                    "lddtfull",
                ],
            )
            structure_name = result_file.name
            # Calculate mean using polars syntax
            averages[structure_name] = df.select(pl.col(score_field).mean()).item()
        except Exception as e:
            logger.warning(f"Failed to process {result_file}: {e}")

    # Sort by score (descending)
    return sorted(averages.items(), key=lambda x: x[1], reverse=True)


def calculate_inertia_matrix(structure_file: Path) -> TransformationMatrix:
    """Calculate rotation matrix using moment of inertia with gemmi directly."""
    # Parse structure with gemmi
    structure = gemmi.read_structure(str(structure_file))

    # Extract CA coordinates
    ca_coords = []

    # Iterate through the structure to get CA atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == "CA":
                        pos = atom.pos
                        ca_coords.append([pos.x, pos.y, pos.z])

    if not ca_coords:
        raise ValueError(f"No CA atoms found in {structure_file}")

    # Convert to numpy array
    coords = np.array(ca_coords)

    # Use unit weights for all atoms
    weights = np.ones(len(coords))

    # Calculate transformation using flatprot's utility function
    return calculate_inertia_transformation(coords, weights)


def create_alignment_database(
    representative_structures: Dict[str, Tuple[str, float]],
    domains_dir: Path,
    output_file: Path,
) -> None:
    """
    Create an HDF5 alignment database with rotation matrices.

    Args:
        representative_structures: Dictionary mapping superfamily IDs to (structure_name, score) tuples
        domains_dir: Directory containing domain PDB files
        output_file: Output HDF5 file path
    """
    # Create alignment database
    db = AlignmentDatabase(output_file)

    with db:
        # Process each representative structure
        for sf_id, (
            original_structure_name,
            score,
        ) in representative_structures.items():
            # Calculate rotation matrix for this structure
            structure_file = domains_dir / f"{sf_id}.pdb"
            try:
                rotation_matrix = calculate_inertia_matrix(structure_file)

                # Create database entry
                entry = AlignmentDBEntry(
                    rotation_matrix=rotation_matrix,
                    entry_id=sf_id,
                    structure_name=sf_id,  # Use SF ID as structure name
                )

                # Add to database
                db.add_entry(entry)
                logger.info(
                    f"Added entry {sf_id} (original: {original_structure_name})"
                )
            except Exception as e:
                logger.warning(f"Failed to process {sf_id}: {e}")
                continue


def save_info_file(
    output_dir: Path,
    representative_structures: Dict[str, Tuple[str, float]],
    domains_dir: Path,
) -> None:
    """
    Save database info TSV file using polars with separate domain information columns.

    Args:
        output_dir: Path to output directory
        representative_structures: Dictionary mapping superfamily IDs to (structure_name, score) tuples
        domains_dir: Directory containing domain PDB files
    """
    # Create a polars DataFrame with separate columns for domain information
    data = {
        "ID": [],  # Superfamily ID
        "representative": [],  # Representative name (superfamily ID)
        "pdb_id": [],  # PDB ID of the original structure
        "chain": [],  # Chain identifier
        "start_res": [],  # Start residue number
        "end_res": [],  # End residue number
        "score": [],  # Alignment score
        "rotation_matrix": [],  # 3x3 rotation matrix as string
        "translation_vector": [],  # 3x1 translation vector as string
        "rotation_type": [],  # Type of rotation (inertia, etc.)
    }

    for sf_id, (structure_name, score) in representative_structures.items():
        # Parse the original structure name to extract components
        # Format is typically: pdb_id_chain_start_res_end_res
        try:
            parts = structure_name.split("_")
            pdb_id = parts[0]
            chain = parts[1]
            start_res = int(parts[2])
            end_res = int(parts[3])
        except (IndexError, ValueError):
            # Handle cases where the structure name doesn't follow the expected format
            logger.warning(f"Could not parse structure name: {structure_name}")
            pdb_id = structure_name
            chain = "?"
            start_res = 0
            end_res = 0

        # Try to calculate the transformation matrix
        structure_file = domains_dir / f"{sf_id}.pdb"
        try:
            # Calculate matrix
            matrix = calculate_inertia_matrix(structure_file)

            # Convert rotation matrix to string
            r = matrix.rotation_matrix
            rotation_str = f"({r[0, 0]}, {r[0, 1]}, {r[0, 2]}, {r[1, 0]}, {r[1, 1]}, {r[1, 2]}, {r[2, 0]}, {r[2, 1]}, {r[2, 2]})"

            # Check if there's a translation component and convert to string
            if (
                hasattr(matrix, "translation_vector")
                and matrix.translation_vector is not None
            ):
                t = matrix.translation_vector
                translation_str = f"({t[0]}, {t[1]}, {t[2]})"
            else:
                translation_str = "(0.0, 0.0, 0.0)"

            rotation_type = "inertia"
        except Exception as e:
            logger.warning(f"Failed to calculate transformation for {sf_id}: {e}")
            # Use identity matrix as fallback
            rotation_str = "(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)"
            translation_str = "(0.0, 0.0, 0.0)"
            rotation_type = "identity"

        data["ID"].append(sf_id)
        data["representative"].append(sf_id)  # Use SF ID as representative name
        data["pdb_id"].append(pdb_id)
        data["chain"].append(chain)
        data["start_res"].append(start_res)
        data["end_res"].append(end_res)
        data["score"].append(score)
        data["rotation_matrix"].append(rotation_str)
        data["translation_vector"].append(translation_str)
        data["rotation_type"].append(rotation_type)

    # Create polars DataFrame
    df = pl.DataFrame(data)

    # Write to TSV file
    df.write_csv(output_dir / "db_info.tsv", separator="\t")
    logger.info(
        f"Saved database info to {output_dir / 'db_info.tsv'} with transformation matrices"
    )


def cleanup_intermediate_files(output_dir: Path, keep_domains: bool = True) -> None:
    """Clean up intermediate files, keeping only essential outputs.

    Args:
        output_dir: Path to the output directory
        keep_domains: Whether to keep the domain PDB files
    """
    logger.info("Cleaning up intermediate files...")

    # Remove FoldSeek database files (not needed after database creation)
    foldseek_dir = output_dir / "foldseek_db"
    if foldseek_dir.exists():
        logger.info(f"Removing FoldSeek database: {foldseek_dir}")
        shutil.rmtree(foldseek_dir)

    # Optionally remove domain files (they can be large but might be useful for reference)
    if not keep_domains:
        domains_dir = output_dir / "domains"
        if domains_dir.exists():
            logger.info(f"Removing domain PDB files: {domains_dir}")
            shutil.rmtree(domains_dir)

    # Remove any other temporary files in the output directory
    # (log files, intermediate results, etc.)
    for item in output_dir.glob("*.tmp"):
        logger.info(f"Removing temporary file: {item}")
        item.unlink()

    logger.info("Cleanup complete. Kept alignment database and info files.")


def extract_unique_pdb_ids(
    superfamilies: Dict[str, List[Tuple[str, str, int, int]]],
) -> List[str]:
    """Extract unique PDB IDs from all superfamilies.

    Args:
        superfamilies: Dictionary mapping superfamily IDs to domain information

    Returns:
        List of unique PDB IDs
    """
    unique_pdb_ids = set()
    for domains in superfamilies.values():
        for pdb_id, _, _, _ in domains:
            unique_pdb_ids.add(pdb_id.lower())
    return list(unique_pdb_ids)


def batch_extract_domains(
    domains_info: List[Tuple[Path, str, int, int, Path]], max_workers: int = 8
) -> List[str]:
    """Extract multiple domains in parallel.

    Args:
        domains_info: List of (pdb_file, chain, start_res, end_res, output_file) tuples
        max_workers: Maximum number of parallel workers

    Returns:
        List of domain names that were successfully extracted
    """

    def process_domain(args):
        pdb_file, chain, start_res, end_res, output_file = args
        domain_name = f"{pdb_file.stem}_{chain}_{start_res}_{end_res}"
        success = extract_domain(pdb_file, chain, start_res, end_res, output_file)
        return domain_name if success else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_domain, domains_info))

    return [domain for domain in results if domain is not None]


def main():
    args = parse_arguments()

    # Create temporary directory if not provided
    if args.tmp_dir is None:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = Path(tmp_dir_obj.name)
    else:
        tmp_dir = args.tmp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir_obj = None

    try:
        # Create output directory
        if args.output.exists():
            if not args.force:
                logger.error(
                    f"Output directory {args.output} already exists. Use --force to overwrite."
                )
                return 1
            shutil.rmtree(args.output)
        args.output.mkdir(parents=True)

        # Get SCOP file
        if args.scop_file is None:
            scop_file = tmp_dir / "scop-cla-latest.txt"
            download_scop_file(scop_file)
        else:
            scop_file = args.scop_file

        # Parse SCOP superfamilies
        superfamilies = get_sf_proteins_domains_region(scop_file)

        # Create working directories
        pdb_dir = tmp_dir / "pdbs"
        domains_dir = args.output / "domains"
        foldseek_dir = args.output / "foldseek_db"
        results_dir = tmp_dir / "results"

        for directory in [pdb_dir, domains_dir, foldseek_dir, results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Extract all unique PDB IDs to download
        pdb_ids_to_download = extract_unique_pdb_ids(superfamilies)
        logger.info(
            f"Found {len(pdb_ids_to_download)} unique PDB structures to download"
        )

        # Download all PDB files in parallel
        logger.info("Starting parallel download of PDB files...")
        pdb_files = asyncio.run(
            batch_download_pdb_files(
                pdb_ids_to_download,
                pdb_dir,
                max_concurrent=20,  # Adjust based on your connection and server limits
            )
        )

        downloaded_count = sum(1 for path in pdb_files.values() if path is not None)
        logger.info(
            f"Successfully downloaded {downloaded_count}/{len(pdb_ids_to_download)} PDB files"
        )

        # Process superfamilies to get representative structures
        representative_structures = {}

        # Process a limited number of superfamilies for testing
        # In production, remove the limit
        superfamily_limit = 5
        for sf_id, domains in list(superfamilies.items())[:superfamily_limit]:
            logger.info(f"Processing superfamily {sf_id} with {len(domains)} domains")

            # Create directory for this superfamily
            sf_dir = tmp_dir / "sf" / sf_id
            sf_domain_dir = sf_dir / "domains"
            sf_results_dir = sf_dir / "results"
            sf_tmp_dir = sf_dir / "tmp"

            for directory in [sf_dir, sf_domain_dir, sf_results_dir, sf_tmp_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            # Prepare domain extraction tasks
            domain_tasks = []
            for pdb_id, chain, start_res, end_res in domains:
                pdb_file = pdb_files.get(pdb_id.lower())
                if pdb_file is None:
                    continue

                domain_name = f"{pdb_id}_{chain}_{start_res}_{end_res}"
                domain_file = sf_domain_dir / f"{domain_name}.pdb"
                domain_tasks.append((pdb_file, chain, start_res, end_res, domain_file))

            # Extract domains in parallel
            valid_domains = batch_extract_domains(domain_tasks)

            # Skip if no valid domains
            if len(valid_domains) == 0:
                logger.warning(f"No valid domains found for {sf_id}")
                continue

            # Create FoldSeek database for this superfamily
            db_path = create_foldseek_database(sf_domain_dir, sf_dir, args.foldseek)

            # Run all-vs-all comparison
            run_foldseek_searches(
                sf_domain_dir, db_path, sf_results_dir, sf_tmp_dir, args.foldseek
            )

            # Find representative structure
            avg_scores = calculate_average_scores(sf_results_dir)
            if len(avg_scores) > 0:
                representative_structures[sf_id] = avg_scores[0]

                # Copy representative structure to output with superfamily ID as name
                original_name = avg_scores[0][0]
                src_file = sf_domain_dir / f"{original_name}.pdb"
                dst_file = domains_dir / f"{sf_id}.pdb"
                shutil.copy(src_file, dst_file)

                # Log selection of representative
                logger.info(
                    f"Selected '{original_name}' as representative for superfamily {sf_id}"
                )

        # Create final FoldSeek database with representatives
        create_foldseek_database(domains_dir, foldseek_dir, args.foldseek)

        # Create alignment database without passing transformation matrices
        db_file = args.output / "alignment.h5"
        create_alignment_database(representative_structures, domains_dir, db_file)

        # Save info file without passing transformation matrices
        save_info_file(args.output, representative_structures, domains_dir)

        logger.info(f"Database creation complete. Output in {args.output}")

        # Clean up intermediate files unless --keep_all flag is set
        if not args.keep_all:
            cleanup_intermediate_files(args.output)

    finally:
        # Clean up temporary directory
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


if __name__ == "__main__":
    sys.exit(main() or 0)
