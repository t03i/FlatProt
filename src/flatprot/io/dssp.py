# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import polars as pl

from flatprot.structure.secondary import SecondaryStructureType


def parse_dssp(dssp_file):
    # Skip header lines and get the line number where data starts
    start_line = 0
    with open(dssp_file) as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                start_line = i + 1
                break

    # Define the fixed-width column specifications
    column_specs = [
        ("residue_number", (0, 5)),
        ("chain", (11, 12)),
        ("amino_acid", (13, 14)),
        ("structure", (16, 17)),
        ("bp1", (26, 29)),
        ("bp2", (29, 32)),
        ("acc", (35, 38)),
    ]

    # Use polars read_csv with fixed-width columns
    df = pl.read_csv(
        dssp_file,
        skip_rows=start_line,
        has_header=False,
        separator=None,  # Fixed-width mode
        columns=column_specs,
        dtypes={
            "residue_number": pl.Int32,
            "chain": pl.Utf8,
            "amino_acid": pl.Utf8,
            "structure": pl.Utf8,
            "bp1": pl.Utf8,
            "bp2": pl.Utf8,
            "acc": pl.Float32,
        },
        truncate_ragged_lines=True,
    )

    return get_secondary_structure_segments(df)


def get_secondary_structure_segments(
    df: pl.DataFrame,
) -> list[tuple[SecondaryStructureType, int, int]]:
    # Map 8-state to 3-state
    # Helix types: H (α-helix), G (3-10 helix), and I (π-helix)
    # Sheet types: E (β-strand) and B (β-bridge)
    changes = df.with_columns(
        [
            pl.col("structure").shift(-1).ne(pl.col("structure")).alias("is_change"),
            # Modified to include all helix and sheet types
            pl.col("structure").is_in(["H", "G", "I", "E", "B"]).alias("is_ss"),
            # New column to map to simplified structure types
            pl.when(pl.col("structure").is_in(["H", "G", "I"]))
            .then(pl.lit("H"))
            .when(pl.col("structure").is_in(["E", "B"]))
            .then(pl.lit("E"))
            .otherwise(pl.lit("C"))
            .alias("simple_structure"),
        ]
    )

    # Get indices where segments start and end
    starts = (
        changes.filter(
            (changes["is_ss"])
            & (
                changes["simple_structure"].shift(1).is_null()
                | changes["simple_structure"].shift(1).ne(changes["simple_structure"])
            )
        )
        .get_column("simple_structure")
        .zip_with(pl.Series(range(len(df))))
    )

    ends = (
        changes.filter(
            (changes["is_ss"])
            & (
                changes["simple_structure"].shift(-1).is_null()
                | changes["simple_structure"].shift(-1).ne(changes["simple_structure"])
            )
        )
        .get_column("simple_structure")
        .zip_with(pl.Series(range(len(df))))
    )

    segments = [
        (
            SecondaryStructureType.HELIX if ss == "H" else SecondaryStructureType.SHEET,
            start,
            end,
        )
        for (ss, start), (_, end) in zip(starts, ends)
    ]

    return segments
