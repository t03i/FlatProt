# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Helper script to generate report on PDB downloads
import os
import json
import polars as pl
import logging
from collections import Counter
from typing import Dict
from pathlib import Path
from snakemake.script import snakemake

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=snakemake.log[0],
)
logger = logging.getLogger(__name__)


def get_download_status(pdb_dir: str) -> Dict[str, bool]:
    """
    Collect download status for all PDB files.

    Args:
        pdb_dir: Directory containing PDB status files

    Returns:
        Dictionary mapping PDB IDs to download success status
    """
    status_dict = {}
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".status.json"):
            pdb_id = filename.split(".")[0]
            with open(os.path.join(pdb_dir, filename), "r") as f:
                try:
                    status_data = json.load(f)
                    status_dict[pdb_id] = status_data.get("success", False)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON in {filename}")
                    status_dict[pdb_id] = False

    return status_dict


def get_missing_structures_by_superfamily(
    superfamilies_df: pl.DataFrame, status_dict: Dict[str, bool]
) -> pl.DataFrame:
    """
    Determine which structures are missing by superfamily.

    Args:
        superfamilies_df: DataFrame with superfamily information
        status_dict: Dictionary with download status for each PDB ID

    Returns:
        DataFrame with count of missing structures by superfamily
    """
    # Add success status to superfamilies dataframe
    with_status = superfamilies_df.with_columns(pl.lit(False).alias("download_success"))

    # Update success status based on status_dict
    rows = []
    for row in with_status.iter_rows(named=True):
        pdb_id = row["pdb_id"]
        success = status_dict.get(pdb_id, False)
        row_dict = dict(row)
        row_dict["download_success"] = success
        rows.append(row_dict)

    updated_df = pl.DataFrame(rows)

    # Group by superfamily and count missing structures
    missing_by_sf = (
        updated_df.filter(~pl.col("download_success"))
        .group_by("sf_id")
        .agg(
            pl.len().alias("missing_count"),
            pl.col("pdb_id").alias("missing_pdbs"),
        )
    )

    # Add total structures count per superfamily
    total_by_sf = superfamilies_df.group_by("sf_id").agg(pl.len().alias("total_count"))

    # Join to get both missing and total counts
    result = missing_by_sf.join(total_by_sf, on="sf_id", how="right")

    # Fill missing values for superfamilies with no missing structures
    result = result.fill_null(0)
    result = result.with_columns(
        pl.when(pl.col("missing_pdbs").is_null())
        .then(pl.lit([]))
        .otherwise(pl.col("missing_pdbs"))
        .alias("missing_pdbs")
    )

    # Calculate missing percentage
    result = result.with_columns(
        (pl.col("missing_count") / pl.col("total_count") * 100).alias("missing_percent")
    )

    return result.sort("missing_count", descending=True)


def generate_rst_report(
    missing_by_sf: pl.DataFrame,
    status_dict: Dict[str, bool],
    output_file: str,
) -> None:
    """
    Generate a reStructuredText report for the PDB download results.

    Args:
        missing_by_sf: DataFrame with missing structures by superfamily
        status_dict: Dictionary with download status for each PDB ID
        output_file: Path to save the RST report
    """
    # Calculate statistics
    total_attempts = len(status_dict)
    succeeded = sum(1 for success in status_dict.values() if success)
    failed = total_attempts - succeeded
    success_rate = (succeeded / total_attempts * 100) if total_attempts > 0 else 0

    # Get top 5 superfamilies with missing structures (where missing > 0)
    top_missing = missing_by_sf.filter(pl.col("missing_count") > 0).head(5)

    # Calculate distribution of missing structures
    missing_counts = missing_by_sf.get_column("missing_count").to_list()
    missing_dist = Counter(missing_counts)
    missing_dist_sorted = sorted(missing_dist.items())

    # Generate the RST report
    with open(output_file, "w") as f:
        f.write("PDB Download Report\n")
        f.write("=================\n\n")

        f.write("Summary\n")
        f.write("-------\n\n")

        f.write(f"* **Total Download Attempts**: {total_attempts}\n")
        f.write(f"* **Successful Downloads**: {succeeded} ({success_rate:.2f}%)\n")
        f.write(f"* **Failed Downloads**: {failed} ({100 - success_rate:.2f}%)\n\n")

        # Top superfamilies with missing structures
        f.write("Top Superfamilies with Missing Structures\n")
        f.write("----------------------------------------\n\n")

        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n\n")
        f.write("   * - Superfamily ID\n")
        f.write("     - Missing Structures\n")
        f.write("     - Total Structures\n")
        f.write("     - Missing Percentage\n")

        for row in top_missing.iter_rows(named=True):
            f.write(f"   * - {row['sf_id']}\n")
            f.write(f"     - {row['missing_count']}\n")
            f.write(f"     - {row['total_count']}\n")
            f.write(f"     - {row['missing_percent']:.2f}%\n")

        # Distribution of missing structure counts
        f.write("\nMissing Structure Distribution\n")
        f.write("----------------------------\n\n")

        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n\n")
        f.write("   * - Missing Structures per Superfamily\n")
        f.write("     - Number of Superfamilies\n")

        for missing_count, sf_count in missing_dist_sorted:
            f.write(f"   * - {missing_count}\n")
            f.write(f"     - {sf_count}\n")


def main() -> None:
    """
    Main processing function for the PDB download report.

    Analyzes download status files, calculates statistics on missing structures,
    and generates a RST report.
    """
    # Get input and output paths from Snakemake
    superfamilies_file = snakemake.input.superfamilies
    pdb_dir = snakemake.params.pdb_dir
    report_output = snakemake.output.report
    flag_file = snakemake.output.flag

    # Load superfamilies data
    logger.info(f"Loading superfamily information from {superfamilies_file}")
    superfamilies_df = pl.read_csv(superfamilies_file, separator="\t")

    # Get download status for all PDBs
    logger.info(f"Analyzing download status in {pdb_dir}")
    status_dict = get_download_status(pdb_dir)

    # Calculate missing structures by superfamily
    missing_by_sf = get_missing_structures_by_superfamily(superfamilies_df, status_dict)

    # Generate report
    logger.info(f"Generating download report at {report_output}")
    generate_rst_report(missing_by_sf, status_dict, report_output)

    logger.info("Download report generation completed")
    Path(flag_file).touch()

    logger.info("Flag file created")


# Run main function with snakemake object
if __name__ == "__main__":
    main()
