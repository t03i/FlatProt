# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import polars as pl
from pathlib import Path
import shutil
import logging
from typing import List, Optional, Tuple

from snakemake.script import snakemake


def read_alignment(alignment_file: Path) -> pl.DataFrame:
    """
    Read alignment file into a Polars DataFrame.

    Args:
        alignment_file: Path to the alignment file

    Returns:
        Polars DataFrame containing alignment information
    """
    try:
        return pl.read_csv(
            alignment_file,
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
            ],
        )
    except Exception as e:
        logging.warning(f"Failed to read alignment file {alignment_file}: {e}")
        return pl.DataFrame(
            schema={
                "query": str,
                "target": str,
                "qstart": int,
                "qend": int,
                "tstart": int,
                "tend": int,
                "tseq": str,
                "prob": float,
                "alntmscore": float,
            }
        )


def combine_alignments(alignment_files: List[Path]) -> pl.DataFrame:
    """
    Combine multiple alignment files into a single DataFrame.

    Args:
        alignment_files: List of alignment file paths

    Returns:
        Combined Polars DataFrame
    """
    if not alignment_files:
        return pl.DataFrame(
            schema={
                "query": str,
                "target": str,
                "qstart": int,
                "qend": int,
                "tstart": int,
                "tend": int,
                "tseq": str,
                "prob": float,
                "alntmscore": float,
            }
        )

    dataframes = [read_alignment(f) for f in alignment_files]
    return pl.concat(dataframes)


def find_best_representative(
    alignment: pl.DataFrame, domain_dir: Path
) -> Optional[Tuple[Path, float]]:
    """
    Find the best representative domain based on alignment scores.

    Args:
        alignment: DataFrame with alignment information
        domain_dir: Directory containing domain PDB files

    Returns:
        Tuple of (path to best domain file, score) or None if no valid domains
    """
    if alignment.is_empty():
        # If no alignments, find any PDB file in the domain directory
        pdb_files = list(domain_dir.glob("*.pdb"))
        if pdb_files:
            return (pdb_files[0], 0.0)
        return None

    # Calculate average TM-score for each domain when it's used as a target
    avg_scores = (
        alignment.group_by("target")
        .agg(pl.mean("alntmscore").alias("avg_tmscore"))
        .sort("avg_tmscore", descending=True)
    )

    # Iterate through domains in order of score
    for row in avg_scores.iter_rows(named=True):
        domain_id = row["target"]
        score = row["avg_tmscore"]

        # Extract components from domain_id (if needed)
        # Assuming domain_id format is pdb_chain_start_end
        filename = f"{domain_id}.pdb"
        domain_file = domain_dir / filename

        if domain_file.exists():
            return (domain_file, score)

    # Fallback if no file is found
    pdb_files = list(domain_dir.glob("*.pdb"))
    if pdb_files:
        return (pdb_files[0], 0.0)

    return None


def main() -> None:
    """
    Main function to find and save the representative domain for a superfamily.
    """
    # Get input and output from Snakemake
    alignment_files = snakemake.input.alignment_files
    domain_dir = Path(snakemake.params.domain_dir)
    output_file = Path(snakemake.output.representative_domain)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Combine all alignment files
    alignment = combine_alignments([Path(f) for f in alignment_files])

    # Find the best representative
    best_domain = find_best_representative(alignment, domain_dir)

    if best_domain:
        domain_file, score = best_domain
        # Copy the best domain file to the output location
        shutil.copy(domain_file, output_file)
        logging.info(
            f"Selected {domain_file.name} as representative domain (score: {score})"
        )
    else:
        # Create an empty file to satisfy Snakemake
        with open(output_file, "w") as f:
            f.write("NO_REPRESENTATIVE_FOUND\n")
        logging.warning(f"No representative domain found for {snakemake.params.sf_id}")


if __name__ == "__main__":
    main()
