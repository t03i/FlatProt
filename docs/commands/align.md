# Align Command

The align command finds the best matching protein superfamily for a given structure and returns the optimal transformation matrix for standardized visualization.

## Usage

```bash
flatprot align STRUCTURE_FILE [options]
```

## Arguments

-   `structure_file`: Path to input protein structure (PDB/mmCIF format)

## Options

-   `-m, --matrix MATRIX_OUT_PATH`: Path to save the transformation matrix (default: "alignment_matrix.npy")
-   `-i, --info INFO_OUT_PATH`: Path to save additional alignment information as JSON (optional)
-   `-f, --foldseek FOLDSEEK_PATH`: Path to FoldSeek executable (default: "foldseek")
-   `-d, --database DATABASE_PATH`: Path to custom alignment database directory
-   `-n, --database-file-name NAME`: Name of the alignment database file (default: "alignments.h5")
-   `-b, --foldseek-db FOLDSEEK_DB_PATH`: Path to custom Foldseek database
-   `-p, --min-probability THRESHOLD`: Minimum alignment probability threshold (default: 0.5)
-   `--download-db`: Force download of the latest database
-   `--quiet`: Suppress all output except errors
-   `--verbose`: Print additional information

### Examples

Basic alignment:

```bash
flatprot align protein.pdb
```

Custom output paths:

```bash
flatprot align protein.pdb -m rotation.npy -i alignment_info.json
```

Using custom database:

```bash
flatprot align protein.pdb --database /path/to/custom/db
```

Adjusting probability threshold:

```bash
flatprot align protein.pdb --min-probability 0.7
```

## Output

The command produces two main outputs:

1. **Transformation Matrix** (NumPy format)

    - A rotation matrix that aligns the input structure to its matched superfamily
    - Used for standardized visualization
    - Saved to the path specified by `--matrix` (default: "alignment_matrix.npy")

2. **Alignment Information** (Optional JSON)

    - Contains detailed alignment results including:

        ```json
        {
            "structure_file": "path/to/input.pdb",
            "matched_family": "sf_1234",
            "probability": 0.85,
            "aligned_region": {
                "start": 1,
                "end": 100
            },
            "message": "Successfully aligned to superfamily sf_1234 with 85% probability"
        }
        ```

    - Only saved if `--info` path is specified

## Error Handling

The command handles several types of errors:

-   Invalid structure files
-   Missing FoldSeek executable
-   No significant alignments found (suggests lowering probability threshold)
-   Database-related issues (suggests using `--download-db`)
-   Output file writing errors
-   FoldSeek execution failures

If no significant alignment is found, try lowering the `--min-probability` threshold. For database issues, using `--download-db` can help ensure you have the latest version.
