# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

# Helper script to parse SCOP file and extract superfamily information
import polars as pl
import logging
from collections import Counter
from typing import List

from snakemake.script import snakemake

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=snakemake.log[0],
)
logger = logging.getLogger(__name__)


def get_sf_proteins_domains_region(scop_file: str) -> pl.DataFrame:
    """
    Extract SCOP superfamily information from SCOP classification file.

    Args:
        scop_file: Path to the SCOP classification file

    Returns:
        Polars DataFrame with superfamily information
    """
    logger.info(f"Parsing SCOP classification file: {scop_file}")

    # Skip header comments and read data
    with open(scop_file, "r") as f:
        header_lines = 0
        for line in f:
            if line.startswith("#"):
                header_lines += 1
            else:
                break

    # Read the file with polars
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
        pl.col("SCOPCLA")
        .str.extract_all(pattern=r"SF=(\d+)")
        .list.get(0)
        .str.replace("SF=", "")
        .alias("sf_id")
    )

    # Process PDB regions to extract chain and residue range
    df = df.with_columns(
        [
            # Extract chain (characters before the colon)
            pl.col("SF_PDBREG").str.extract(r"([A-Za-z0-9]+):").alias("chain"),
            # Extract start residue (number between colon and dash)
            pl.col("SF_PDBREG")
            .str.extract(r":(-?\d+)-")
            .cast(pl.Int32)
            .alias("start_res"),
            # Extract end residue (last number in the string)
            pl.col("SF_PDBREG")
            .str.extract(r"-(-?\d+)$")
            .cast(pl.Int32)
            .alias("end_res"),
        ]
    )

    # Convert to lowercase for consistency
    df = df.with_columns(pl.col("SF_PDBID").str.to_lowercase().alias("pdb_id"))

    # Select relevant columns
    result_df = df.select(["sf_id", "pdb_id", "chain", "start_res", "end_res"])

    return result_df


def generate_rst_report(
    df: pl.DataFrame,
    unique_pdbs: List[str],
    output_file: str,
) -> None:
    """
    Generate a reStructuredText report for the SCOP parsing results.

    Args:
        df: Polars DataFrame with superfamily information
        unique_pdbs: List of unique PDB IDs
        output_file: Path to save the RST report
    """
    # Calculate statistics
    sf_count = df.get_column("sf_id").n_unique()
    pdb_count = len(unique_pdbs)
    domain_count = df.height

    # Get invalid entries
    invalid_entries = df.filter(
        (pl.col("start_res").is_null()) | (pl.col("end_res").is_null())
    ).select(["sf_id", "pdb_id", "chain", "start_res", "end_res"])
    invalid_count = invalid_entries.height

    # Count domains per superfamily
    sf_domain_counts = df.group_by("sf_id").agg(pl.len()).sort("len", descending=True)
    top_sf = sf_domain_counts.head(10)

    # Calculate average domains per superfamily
    avg_domains_per_sf = domain_count / sf_count if sf_count > 0 else 0

    # Generate the RST report
    with open(output_file, "w") as f:
        f.write("SCOP Parsing Report\n")
        f.write("==================\n\n")

        f.write("Summary\n")
        f.write("-------\n\n")

        f.write(f"* **Total Superfamilies**: {sf_count}\n")
        f.write(f"* **Total PDB Entries**: {pdb_count}\n")
        f.write(f"* **Total Domains**: {domain_count}\n")
        f.write(f"* **Invalid Residue Ranges**: {invalid_count}\n")
        f.write(f"* **Average Domains per Superfamily**: {avg_domains_per_sf:.2f}\n\n")

        # Add invalid entries section if any exist
        if invalid_count > 0:
            f.write("Invalid Residue Ranges\n")
            f.write("--------------------\n\n")
            f.write(".. list-table::\n")
            f.write("   :header-rows: 1\n\n")
            f.write("   * - Superfamily ID\n")
            f.write("     - PDB ID\n")
            f.write("     - Region\n")

            for row in invalid_entries.iter_rows():
                f.write(f"   * - {row[0]}\n")
                f.write(f"     - {row[1]}\n")
                f.write(f"     - {row[2]}\n")

            f.write("\n")

        # Top superfamilies by domain count
        f.write("Top Superfamilies\n")
        f.write("---------------\n\n")

        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n\n")
        f.write("   * - Superfamily ID\n")
        f.write("     - Domain Count\n")

        for row in top_sf.iter_rows():
            f.write(f"   * - {row[0]}\n")
            f.write(f"     - {row[1]}\n")

        # Distribution of domain counts
        domain_dist = Counter(sf_domain_counts.get_column("len").to_list())
        domain_dist_sorted = sorted(domain_dist.items())

        f.write("\nDomain Count Distribution\n")
        f.write("----------------------\n\n")

        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n\n")
        f.write("   * - Domains per Superfamily\n")
        f.write("     - Number of Superfamilies\n")

        for domains, count in domain_dist_sorted:
            f.write(f"   * - {domains}\n")
            f.write(f"     - {count}\n")


# Main processing function
def main() -> None:
    """
    Main processing function for the SCOP parsing workflow.

    Reads SCOP classification file, extracts superfamily information,
    and generates output files including a RST report.
    """
    scop_file = snakemake.input.scop_file
    superfamilies_output = snakemake.output.superfamilies
    report_output = snakemake.output.report

    # Parse SCOP file
    df = get_sf_proteins_domains_region(scop_file)

    # --- Filtering for Invalid/Complex Regions ---
    initial_rows = df.height
    logger.info(f"Initial rows parsed from SCOP file: {initial_rows}")

    # Filter out rows with commas in PDB region (discontinuous domains)
    df_no_comma = df.filter(~pl.col("SF_PDBREG").str.contains(","))
    comma_dropped_rows = initial_rows - df_no_comma.height
    if comma_dropped_rows > 0:
        logger.info(
            f"Dropped {comma_dropped_rows} rows due to commas in SF_PDBREG (discontinuous domains)."
        )

    # Filter out entries with null start/end values AFTER basic extraction
    # (These might occur even without commas due to parsing errors or bad input)
    df_filtered = df_no_comma.filter(
        (pl.col("start_res").is_not_null()) & (pl.col("end_res").is_not_null())
    )
    null_dropped_rows = df_no_comma.height - df_filtered.height
    if null_dropped_rows > 0:
        logger.info(
            f"Dropped {null_dropped_rows} additional rows due to null start/end residue after extraction."
        )

    total_dropped = initial_rows - df_filtered.height
    logger.info(
        f"Total rows dropped: {total_dropped}. Final rows for output: {df_filtered.height}"
    )

    # Generate RST report before filtering
    generate_rst_report(df, df.get_column("pdb_id").unique().to_list(), report_output)

    # Save filtered superfamilies information
    # Ensure only the essential columns are saved
    df_to_save = df_filtered.select(
        ["sf_id", "pdb_id", "chain", "start_res", "end_res"]
    )
    df_to_save.write_csv(superfamilies_output, separator="\t")
    logger.info(f"Saved superfamily information to {superfamilies_output}")


# Run main function with snakemake object
if __name__ == "__main__":
    main()
