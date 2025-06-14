#!/usr/bin/env python3
"""
FlatProt Benchmark Structure Setup Script

This script downloads protein structures from the AlphaFold Database based on
a human proteome TSV file for benchmarking FlatProt performance.

USAGE:
    python scripts/runtime_benchmark_setup.py human_proteome.tsv [options]

QUICK START:
    # Download first 100 structures from UniProt human proteome
    python scripts/runtime_benchmark_setup.py human_proteome.tsv \\
        --output-dir data/benchmark_structures \\
        --max-entries 100

    # Download all structures (may take hours and GB of space)
    python scripts/runtime_benchmark_setup.py human_proteome.tsv \\
        --output-dir data/benchmark_structures

INPUT FORMAT:
    TSV file with columns including 'Entry' (UniProt ID) and optionally other metadata.
    The script will parse UniProt IDs from the 'Entry' column.

    Example TSV format:
    Entry	Protein names	Gene Names (primary)	Length	Organism
    P69905	Hemoglobin subunit alpha	HBA1	142	Homo sapiens
    P01112	GTPase HRas	HRAS	189	Homo sapiens

    You can download human proteome data from UniProt:
    1. Go to https://www.uniprot.org/
    2. Search for "organism:human AND reviewed:yes"
    3. Download results as TSV with desired columns

OPTIONS:
    --output-dir PATH      Directory to save structures (default: data/benchmark_structures)
    --max-entries INT      Maximum entries to process (default: all)
    --batch-size INT       Download batch size (default: 50)
    --skip-existing        Skip downloading existing files
    --dry-run             Show what would be downloaded without downloading

OUTPUT FILES:
    - AF-{UNIPROT_ID}-F1-model_v4.cif: Downloaded structure files
    - benchmark_metadata.tsv: Metadata about downloaded structures

EXAMPLE WORKFLOW:
    # 1. Download human proteome TSV from UniProt
    wget -O human_proteome.tsv "https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=organism_id:9606+AND+reviewed:true"

    # 2. Setup benchmark with 200 structures
    python scripts/runtime_benchmark_setup.py human_proteome.tsv \\
        --output-dir data/human_proteome_200 \\
        --max-entries 200

PERFORMANCE NOTES:
    - Downloading 1000 structures takes ~30-60 minutes and ~2-5 GB disk space
    - Some UniProt IDs may not have AlphaFold structures (404 errors are normal)
    - Uses exponential backoff retry logic for failed downloads

REQUIREMENTS:
    - Internet connection for downloading from AlphaFold Database
    - requests library for HTTP downloads
"""

import sys
import csv
import requests
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def parse_proteome_tsv(
    tsv_file: Path, max_entries: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Parse human proteome TSV file and extract UniProt IDs with metadata.

    Args:
        tsv_file: Path to the TSV file
        max_entries: Maximum number of entries to process (None for all)

    Returns:
        List of dictionaries containing protein information
    """
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")

    proteins = []

    try:
        with open(tsv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            # Validate required columns
            if "Entry" not in reader.fieldnames:
                raise ValueError(f"Required column 'Entry' not found in {tsv_file}")

            log.info(f"Available columns: {reader.fieldnames}")

            for i, row in enumerate(reader):
                if max_entries and i >= max_entries:
                    break

                uniprot_id = row.get("Entry", "").strip()
                if not uniprot_id:
                    continue

                protein_info = {
                    "uniprot_id": uniprot_id,
                    "protein_name": row.get("Protein names", "").strip(),
                    "gene_name": row.get("Gene Names (primary)", "").strip(),
                    "length": row.get("Length", "0").strip(),
                    "organism": row.get("Organism", "").strip(),
                }

                proteins.append(protein_info)

        log.info(f"Parsed {len(proteins)} protein entries from {tsv_file}")
        return proteins

    except Exception as e:
        log.error(f"Failed to parse TSV file {tsv_file}: {e}")
        raise


def download_alphafold_structure(
    uniprot_id: str, output_dir: Path, retry_count: int = 3
) -> Optional[Path]:
    """
    Download AlphaFold structure for a given UniProt ID with retry logic.

    Args:
        uniprot_id: UniProt identifier
        output_dir: Directory to save the structure file
        retry_count: Number of retry attempts

    Returns:
        Path to downloaded file or None if failed
    """
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    output_file = output_dir / f"AF-{uniprot_id}-F1-model_v4.cif"

    # Check if file already exists and is valid
    if output_file.exists() and output_file.stat().st_size > 1000:  # At least 1KB
        log.debug(f"Structure already exists: {output_file}")
        return output_file

    for attempt in range(retry_count):
        try:
            log.info(
                f"Downloading {uniprot_id} (attempt {attempt + 1}/{retry_count})..."
            )

            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()

            # Create output directory if needed
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download with progress indication for large files
            downloaded_size = 0

            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            # Validate downloaded file
            if output_file.stat().st_size == 0:
                log.warning(f"Downloaded file for {uniprot_id} is empty")
                output_file.unlink(missing_ok=True)
                continue

            if output_file.stat().st_size < 1000:  # Suspiciously small
                log.warning(
                    f"Downloaded file for {uniprot_id} is unusually small ({output_file.stat().st_size} bytes)"
                )
                output_file.unlink(missing_ok=True)
                continue

            log.info(
                f"Successfully downloaded {uniprot_id} ({output_file.stat().st_size} bytes)"
            )
            return output_file

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                log.warning(f"Structure not found in AlphaFold DB: {uniprot_id}")
                return None
            else:
                log.warning(f"HTTP error downloading {uniprot_id}: {e}")
        except requests.exceptions.RequestException as e:
            log.warning(f"Request error downloading {uniprot_id}: {e}")
        except Exception as e:
            log.warning(f"Unexpected error downloading {uniprot_id}: {e}")

        if attempt < retry_count - 1:
            time.sleep(2**attempt)  # Exponential backoff

    log.error(f"Failed to download {uniprot_id} after {retry_count} attempts")
    return None


def save_metadata(
    proteins: List[Dict[str, str]], successful_downloads: List[str], output_dir: Path
) -> None:
    """Save metadata about downloaded structures."""
    metadata_file = output_dir / "benchmark_metadata.tsv"

    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        # Header
        writer.writerow(
            [
                "uniprot_id",
                "protein_name",
                "gene_name",
                "length",
                "organism",
                "downloaded",
                "structure_file",
                "download_timestamp",
            ]
        )

        timestamp = datetime.now().isoformat()

        for protein in proteins:
            uniprot_id = protein["uniprot_id"]
            downloaded = uniprot_id in successful_downloads
            structure_file = f"AF-{uniprot_id}-F1-model_v4.cif" if downloaded else ""

            writer.writerow(
                [
                    uniprot_id,
                    protein["protein_name"],
                    protein["gene_name"],
                    protein["length"],
                    protein["organism"],
                    downloaded,
                    structure_file,
                    timestamp if downloaded else "",
                ]
            )

    log.info(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup benchmark structures from human proteome TSV"
    )
    parser.add_argument("tsv_file", type=Path, help="Path to human proteome TSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark_structures"),
        help="Directory to save structures (default: data/benchmark_structures)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of structures to download per batch (default: 50)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading if structure already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse TSV and show what would be downloaded without downloading",
    )

    args = parser.parse_args()

    log.info("=== FlatProt Benchmark Structure Setup ===")
    log.info(f"TSV file: {args.tsv_file}")
    log.info(f"Output directory: {args.output_dir}")

    # Parse proteome TSV file
    try:
        proteins = parse_proteome_tsv(args.tsv_file, args.max_entries)
    except Exception as e:
        log.error(f"Failed to parse TSV file: {e}")
        return 1

    if not proteins:
        log.error("No proteins found in TSV file")
        return 1

    log.info(f"Found {len(proteins)} proteins to process")

    # Show sample of what will be processed
    log.info("Sample entries:")
    for i, protein in enumerate(proteins[:5]):
        log.info(
            f"  {i+1}. {protein['uniprot_id']} - {protein['protein_name'][:50]}..."
        )

    if args.dry_run:
        log.info("Dry run mode - no structures will be downloaded")
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download structures in batches
    successful_downloads = []
    failed_downloads = []

    total_proteins = len(proteins)
    batch_size = args.batch_size

    for batch_start in range(0, total_proteins, batch_size):
        batch_end = min(batch_start + batch_size, total_proteins)
        batch_proteins = proteins[batch_start:batch_end]

        log.info(
            f"Processing batch {batch_start//batch_size + 1} "
            f"({batch_start + 1}-{batch_end} of {total_proteins})"
        )

        for i, protein in enumerate(batch_proteins):
            uniprot_id = protein["uniprot_id"]
            current_num = batch_start + i + 1

            log.info(f"[{current_num}/{total_proteins}] Processing {uniprot_id}")

            # Skip if already exists and skip-existing is enabled
            existing_file = args.output_dir / f"AF-{uniprot_id}-F1-model_v4.cif"
            if (
                args.skip_existing
                and existing_file.exists()
                and existing_file.stat().st_size > 1000
            ):
                log.info(f"Skipping {uniprot_id} - file already exists")
                successful_downloads.append(uniprot_id)
                continue

            # Download structure
            result = download_alphafold_structure(uniprot_id, args.output_dir)

            if result:
                successful_downloads.append(uniprot_id)
            else:
                failed_downloads.append(uniprot_id)

        # Brief pause between batches
        if batch_end < total_proteins:
            log.info("Completed batch. Pausing briefly...")
            time.sleep(1)

    # Save metadata
    save_metadata(proteins, successful_downloads, args.output_dir)

    # Print summary
    log.info("=== DOWNLOAD SUMMARY ===")
    log.info(f"Total proteins processed: {len(proteins)}")
    log.info(f"Successful downloads: {len(successful_downloads)}")
    log.info(f"Failed downloads: {len(failed_downloads)}")
    log.info(f"Success rate: {len(successful_downloads)/len(proteins)*100:.1f}%")

    if failed_downloads:
        log.info(f"Failed UniProt IDs: {', '.join(failed_downloads[:10])}")
        if len(failed_downloads) > 10:
            log.info(f"... and {len(failed_downloads) - 10} more")

    # Calculate total size
    total_size = 0
    for uniprot_id in successful_downloads:
        structure_file = args.output_dir / f"AF-{uniprot_id}-F1-model_v4.cif"
        if structure_file.exists():
            total_size += structure_file.stat().st_size

    log.info(f"Total downloaded size: {total_size / (1024*1024):.1f} MB")
    log.info(f"Structures saved in: {args.output_dir}")

    return 0 if successful_downloads else 1


if __name__ == "__main__":
    sys.exit(main())
