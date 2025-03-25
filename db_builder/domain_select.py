# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import polars as pl
from pathlib import Path
import shutil
import logging
from typing import Optional, Tuple

from snakemake.script import snakemake

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=snakemake.log[0],
)
logger = logging.getLogger(__name__)


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


def find_best_representative(alignment: pl.DataFrame) -> Optional[Tuple[str, float]]:
    """
    Find the best representative domain based on alignment scores.

    Args:
        alignment: DataFrame with alignment information

    Returns:
        Tuple of (best domain filename, score) or None if no valid alignments
    """
    if alignment.is_empty():
        return None

    # Calculate average TM-score for each domain when it's used as a target
    avg_scores = (
        alignment.group_by("target")
        .agg(pl.mean("alntmscore").alias("avg_tmscore"))
        .sort("avg_tmscore", descending=True)
    )

    if avg_scores.is_empty():
        return None

    # Get the best scoring domain
    best_domain = avg_scores.row(0, named=True)
    return (best_domain["target"], best_domain["avg_tmscore"])


def main() -> None:
    """
    Main function to find and save the representative domain for a superfamily.
    """
    # Get input and output from Snakemake
    alignment_file = Path(snakemake.input.alignment_file)
    domain_dir = Path(snakemake.params.domain_dir)
    output_file = Path(snakemake.output.representative_domain)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read the single alignment file
    alignment = read_alignment(alignment_file)

    # Find the best representative
    best_domain = find_best_representative(alignment)

    if best_domain:
        domain_name, score = best_domain
        source_file = domain_dir / f"{domain_name}.cif"
        if source_file.exists():
            # Copy the best domain file to the output location
            shutil.copy(source_file, output_file)
            logging.info(
                f"Selected {domain_name} as representative domain (score: {score})"
            )
        else:
            logging.error(f"Best domain file {domain_name} not found in {domain_dir}")
    else:
        logging.warning("No valid domains found in alignment")


if __name__ == "__main__":
    main()
