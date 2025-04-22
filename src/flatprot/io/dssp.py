import polars as pl

from flatprot.core import SecondaryStructureType


def parse_dssp(dssp_file):
    # Skip header lines and get the line number where data starts
    start_line = 0
    with open(dssp_file) as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("#  RESIDUE AA"):
                start_line = i + 1
                break

    # Read file as single column
    df = pl.read_csv(
        dssp_file,
        skip_rows=start_line,
        separator="\n",
        has_header=False,
        new_columns=["data"],
    )

    # Comprehensive regex pattern matching the C++ format string
    # cf. https://github.com/cmbi/dssp/blob/42f0758d9a4a4d63e5b0afadfa9bad81b9a277b2/src/dssp.cpp#L55
    pattern = r"""(?x)
        (?P<dssp_num>.{5})           # DSSP number (%.5s)
        (?P<pdb_num>.{5})            # PDB residue number (%.5s)
        (?P<icode>.{1})              # Insertion code (%1.1s)
        (?P<chain>.{1})\s            # Chain ID (%1.1s)
        (?P<aa>.{1})\s{2}           # Amino acid (%c)
        (?P<structure>.{1})        # Secondary structure (%c)
        (?P<structure_details>.{1}) # Secondary structure details (%c)
        (?P<h3>.{1})                 # 3-turn helix (%c)
        (?P<h4>.{1})                 # 4-turn helix (%c)
        (?P<h5>.{1})                 # 5-turn helix (%c)
        (?P<bend>.{1})               # Geometric bend (%c)
        (?P<chirality>.{1})          # Chirality (%c)
        (?P<bridge1>.{1})            # Bridge partner 1 (%c)
        (?P<bridge2>.{1})            # Bridge partner 2 (%c)
        (?P<bp1>.{4})               # Beta bridge partner 1 (%.4s)
        (?P<bp2>.{4})               # Beta bridge partner 2 (%.4s)
        (?P<sheet>.{1})              # Sheet label (%c)
        (?P<acc>.{4})\s             # Accessibility (%.4s)
        (?P<nh_o1>.{11})            # NH-->O 1 (%11s)
        (?P<o_hn1>.{11})            # O-->HN 1 (%11s)
        (?P<nh_o2>.{11})            # NH-->O 2 (%11s)
        (?P<o_hn2>.{11})\s{2}       # O-->HN 2 (%11s)
        (?P<tco>.{6})               # TCO (%6.3f)
        (?P<kappa>.{6})             # Kappa (%6.1f)
        (?P<alpha>.{6})             # Alpha (%6.1f)
        (?P<phi>.{6})               # Phi (%6.1f)
        (?P<psi>.{6})\s             # Psi (%6.1f)
        (?P<x_ca>.{6})\s            # X-CA (%6.1f)
        (?P<y_ca>.{6})\s            # Y-CA (%6.1f)
        (?P<z_ca>.{6})              # Z-CA (%6.1f)
        (?:\s{13}(?P<chain_id>.{4}))? # Optional chain ID (%4.4s)
        (?:\s{6}(?P<auth_chain>.{4}))? # Optional auth chain (%4.4s)
        (?:\s+(?P<ext_num>.{10}))?   # Optional extended number (%10s)
        (?:\s+(?P<ext_resnum>.{10}))? # Optional extended resnum (%10s)
        .*                           # Any remaining characters
    """

    # Parse all fields directly with the regex using named groups cf. https://stackoverflow.com/a/78545671
    df = (
        df.select(pl.col("data").str.extract_groups(pattern))
        .unnest("data")
        .with_columns(
            [
                pl.col("dssp_num", "pdb_num").str.strip_chars(characters=" ").cast(int),
                pl.col(
                    "acc", "tco", "kappa", "alpha", "phi", "psi", "x_ca", "y_ca", "z_ca"
                )
                .str.strip_chars(characters=" ")
                .cast(pl.Float32),
                pl.col("chain_id", "auth_chain")
                .str.strip_chars(characters=" ")
                .cast(str),
                pl.col("ext_num", "ext_resnum")
                .str.strip_chars(characters=" ")
                .cast(int),
            ]
        )
        .with_columns(
            pl.col(col)
            .str.strip_chars(" ")
            .str.split(",")
            .list.to_struct(fields=[f"{col}_resnum", f"{col}_energy"])
            for col in ["nh_o1", "o_hn1", "nh_o2", "o_hn2"]
        )
        .unnest("nh_o1", "o_hn1", "nh_o2", "o_hn2")
    )

    return get_secondary_structure_segments(df)


def get_secondary_structure_segments(
    df: pl.DataFrame,
) -> list[tuple[SecondaryStructureType, int, int]]:
    """
    Extract secondary structure segments from DSSP data.

    Parameters:
        df (pl.DataFrame): DataFrame containing DSSP data with 'structure' column

    Returns:
        list[tuple[SecondaryStructureType, int, int]]: List of tuples containing
            (structure_type, start_position, end_position)
    """
    # Map 8-state to 3-state
    # Helix types: H (α-helix), G (3-10 helix), and I (π-helix)
    # Sheet types: E (β-strand) and B (β-bridge)

    helix_types = pl.Series(["H", "G", "I", "P"])
    sheet_types = pl.Series(["E", "B"])

    changes = df.with_columns(
        [
            pl.col("structure")
            .str.contains_any(pl.concat([helix_types, sheet_types]))
            .alias("is_ss"),
            pl.when(pl.col("structure").str.contains_any(helix_types))
            .then(pl.lit(SecondaryStructureType.HELIX.value))
            .when(pl.col("structure").str.contains_any(sheet_types))
            .then(pl.lit(SecondaryStructureType.SHEET.value))
            .otherwise(pl.lit(SecondaryStructureType.COIL.value))
            .alias("simple_structure"),
        ]
    )
    changes = changes.with_columns(
        [
            pl.col("structure").ne(pl.col("structure").shift(-1)).alias("is_change"),
        ]
    )

    # Get indices where segments start and end
    starts = changes.filter(
        (changes["is_ss"]) & (changes["is_change"].shift(1))
    ).select("simple_structure", "pdb_num")

    ends = changes.filter((changes["is_ss"]) & (changes["is_change"])).select(
        "simple_structure", "pdb_num"
    )
    # Create segments from starts and ends
    segments = [
        (SecondaryStructureType(ss_type), start, end)
        for (ss_type, start), (_, end) in zip(starts.iter_rows(), ends.iter_rows())
    ]
    return segments
