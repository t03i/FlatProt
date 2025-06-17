# Align Command

Find the best matching protein superfamily for a structure using FoldSeek and retrieve the corresponding transformation matrix for standardized visualization.

## Usage

```bash
flatprot align STRUCTURE_FILE [OPTIONS]
```

## Parameters

### Required
- `STRUCTURE_FILE` - Path to the input protein structure file (PDB or mmCIF format)

### Output Options
- `--matrix` / `-m` - Path to save the output transformation matrix (NumPy format) [default: alignment_matrix.npy]
- `--info` / `-i` - Path to save detailed alignment information as JSON. If omitted, info is printed to stdout

### Database Options
- `--foldseek` / `-f` - Path to the FoldSeek executable [default: foldseek]
- `--foldseek-db` / `-b` - Path to a custom FoldSeek database
- `--database` / `-d` - Path to the directory containing the custom FlatProt alignment database (HDF5 file)
- `--database-file-name` / `-n` - Name of the alignment database file [default: alignments.h5]
- `--download-db` - Force download/update of the default database

### Alignment Options
- `--min-probability` / `-p` - Minimum FoldSeek alignment probability threshold (0.0-1.0) [default: 0.5]
- `--target-db-id` - Force alignment against a specific target ID, bypassing probability checks
- `--alignment-mode` / `-a` - Alignment mode [default: family-identity]
  - `family-identity`: Uses direct alignment matrix
  - `family-inertia`: Uses database reference matrix combined with alignment

### General Options
- `--quiet` - Suppress all informational output except errors
- `--verbose` - Print additional debug information during execution

## Alignment Modes

### Family-Identity (Default)
Uses the direct alignment matrix resulting from FoldSeek alignment between input and best-matching database entry.

### Family-Inertia
Uses the pre-calculated reference matrix for the matched superfamily, combined with the FoldSeek alignment matrix. This spreads out structure elements while keeping basic family rotation but sacrifices consistent structure element placement.

## Examples

### Basic Usage
```bash
# Basic alignment (saves to alignment_matrix.npy)
flatprot align protein.pdb

# Specify output paths
flatprot align protein.cif -m rotation.npy -i alignment_info.json
```

### Alignment Modes
```bash
# Family identity alignment (default)
flatprot align protein.pdb --alignment-mode family-identity

# Family inertia alignment
flatprot align protein.pdb --alignment-mode family-inertia
```

### Custom Database and Thresholds
```bash
# Use custom FoldSeek executable and database
flatprot align protein.pdb -f /path/to/foldseek -b /path/to/foldseek_db

# Adjust probability threshold
flatprot align protein.pdb --min-probability 0.7

# Force alignment to specific target
flatprot align protein.pdb --target-db-id "3000114"
```

### Database Management
```bash
# Force database download/update
flatprot align protein.pdb --download-db
```

## Output Files

### Transformation Matrix
A 4x4 NumPy array (`.npy` file) representing the rotation matrix that aligns the input structure. The specific matrix depends on the alignment mode:
- **Family-identity:** Direct FoldSeek alignment matrix
- **Family-inertia:** Pre-calculated reference matrix combined with alignment

This matrix is intended for use with the `flatprot project` command.

### Alignment Information (JSON)
Contains detailed alignment results including:

```json
{
    "structure_file": "path/to/protein.pdb",
    "foldseek_db_path": "/path/to/foldseek_db",
    "min_probability": 0.5,
    "target_db_id": null,
    "alignment_mode": "family-identity",
    "best_hit": {
        "query_id": "protein",
        "target_id": "T1084-D1",
        "probability": 0.995,
        "e_value": 8.74e-15,
        "tm_score": 0.789
    },
    "matrix_file": "rotation.npy"
}
```

## Troubleshooting

### Common Issues

**Invalid or non-existent structure files:**
- Verify the structure file path is correct
- Ensure file is in valid PDB or mmCIF format

**FoldSeek executable not found:**
- Check FoldSeek installation: `which foldseek`
- Specify custom path with `--foldseek /path/to/foldseek`

**No significant alignment found:**
- Lower the probability threshold: `--min-probability 0.3`
- Check if the protein family is represented in the database

**Database corruption or access issues:**
- Try re-downloading: `--download-db`
- Verify database file permissions and accessibility

**Network connectivity issues:**
- Check internet connection for database download
- Consider using a local database with `--foldseek-db`
