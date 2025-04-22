# Align Command

The `align` command finds the best matching protein superfamily for a given structure using Foldseek and retrieves the corresponding pre-calculated transformation matrix for standardized visualization.

## Usage

```bash
flatprot align STRUCTURE_FILE [OPTIONS]
```

## Arguments

-   `STRUCTURE_FILE`: Path to the input protein structure file (PDB or mmCIF format). The file must exist and be readable.

## Options

| Option                 | Short | Type                                | Default                | Description                                                                                                              |
| ---------------------- | ----- | ----------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--matrix`             | `-m`  | Path                                | `alignment_matrix.npy` | Path to save the output transformation matrix (NumPy format).                                                            |
| `--info`               | `-i`  | Path                                | _None_                 | Optional path to save detailed alignment information as JSON. If omitted, info is printed to stdout.                     |
| `--foldseek`           | `-f`  | String                              | `"foldseek"`           | Path to the FoldSeek executable. Assumes it's in PATH if not specified.                                                  |
| `--foldseek-db`        | `-b`  | Path                                | _Default_              | Path to a custom Foldseek database. Defaults to the database bundled with FlatProt or previously downloaded.             |
| `--database`           | `-d`  | Path                                | _Default_              | Path to the directory containing the custom FlatProt alignment database (HDF5 file and supporting files).                |
| `--database-file-name` | `-n`  | String                              | `"alignments.h5"`      | Name of the alignment database file within the `--database` directory.                                                   |
| `--min-probability`    | `-p`  | Float                               | `0.5`                  | Minimum Foldseek alignment probability threshold (0.0 to 1.0).                                                           |
| `--target-db-id`       |       | String                              | _None_                 | Force alignment against a specific target ID from the Foldseek database, bypassing probability checks.                   |
| `--alignment-mode`     | `-a`  | `family-identity`, `family-inertia` | `family-identity`      | Alignment mode: `family-identity` uses the direct alignment matrix, `family-inertia` uses the database reference matrix. |
| `--download-db`        |       | Boolean                             | `false`                | Force download/update of the default database even if it exists locally.                                                 |
| `--quiet`              |       | Boolean                             | `false`                | Suppress all informational output except errors.                                                                         |
| `--verbose`            |       | Boolean                             | `false`                | Print additional debug information during execution.                                                                     |

## Examples

**Basic Alignment (Family Identity):** (Saves direct alignment matrix to `alignment_matrix.npy`)

```bash
flatprot align protein.pdb
```

**Alignment Using Family Inertia:**

In addition to applying the family rotation, it applies the inertia transformation that was calulated for the family.

**Note**: This will spread out structure elements while keeping the basic family rotation but sacrifices consistent structure element placement.

```bash
flatprot align protein.pdb --alignment-mode family-inertia
```

**Specify Output Paths:**

```bash
flatprot align protein.cif -m rotation.npy -i alignment_info.json
```

**Use Custom Foldseek Executable and Database:**

```bash
flatprot align protein.pdb -f /path/to/foldseek -b /path/to/foldseek_db
```

**Adjust Probability Threshold:**

```bash
flatprot align protein.pdb --min-probability 0.7
```

**Force Alignment to a Specific Target:**

```bash
flatprot align protein.pdb --target-db-id "3000114"
```

## Output Files

1.  **Transformation Matrix (`--matrix`):**

    -   A 4x4 NumPy array (`.npy` file) representing the rotation matrix that aligns the input structure.
    -   If `--alignment-mode` is `family-identity` (default), this is the matrix directly resulting from the Foldseek alignment between the input and the best-matching database entry.
    -   If `--alignment-mode` is `family-inertia`, this is the pre-calculated reference matrix for the matched superfamily, combined with the Foldseek alignment matrix.
    -   This matrix is intended for use with the `flatprot project` command.

2.  **Alignment Information (`--info`, Optional JSON):**

    -   If a path is provided via `--info`, a JSON file containing detailed alignment results is saved. The exact content may vary, but typically includes:

        ```json
        {
            "structure_file": "path/to/protein.pdb",
            "foldseek_db_path": "/path/to/foldseek_db",
            "min_probability": 0.5,
            "target_db_id": null,
            "alignment_mode": "family-identity", // Reflects the chosen mode
            "best_hit": {
                "query_id": "protein",
                "target_id": "T1084-D1",
                "probability": 0.995,
                "e_value": 8.74e-15,
                "tm_score": 0.789
            },
            "db_entry": { // Only present if mode is family-inertia
                "entry_id": "T1084-D1",
                "structure_name": "Example Superfamily",
                "rotation_matrix": { ... } // Details of the database matrix
            },
            "matrix_file": "rotation.npy"
        }
        ```

## Error Handling

The command includes robust error handling:

-   **Input Errors:** Checks for invalid or non-existent structure files (`InvalidStructureError`, `FileNotFoundError`).
-   **FoldSeek Errors:** Verifies the FoldSeek executable path. Catches errors during FoldSeek execution (`RuntimeError`, `subprocess.SubprocessError`).
-   **Database Errors:** Handles issues finding or accessing the alignment database (`DatabaseEntryNotFoundError`). Suggests using `--download-db` if corruption is suspected. This error is more likely in `family-inertia` mode.
-   **Alignment Quality:** If no alignment meets the `--min-probability` threshold, it reports `NoSignificantAlignmentError` and suggests lowering the threshold.
-   **Output Errors:** Catches errors writing the matrix or info files (`OutputFileError`).
-   **General Errors:** Catches other `FlatProtError` types and unexpected exceptions.

Exit codes are `0` for success and `1` for any error.
