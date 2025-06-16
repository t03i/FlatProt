# Project Command

The `project` command transforms a 3D protein structure into a standardized 2D SVG representation, optionally applying custom transformations, styles, and annotations.

## Usage

```bash
flatprot project STRUCTURE_FILE [OPTIONS]
```

## Arguments

-   `STRUCTURE_FILE`: Path to the input protein structure file (PDB or mmCIF format). Required.

## Options

| Option            | Short | Type    | Default  | Description                                                                                                                                     |
| ----------------- | ----- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output`        | `-o`  | Path    | _stdout_ | Path to save the output SVG file. If omitted, the SVG content is printed to standard output.                                                    |
| `--matrix`        |       | Path    | _None_   | Path to a custom transformation matrix (NumPy `.npy` format). If omitted, an inertia-based transformation is applied. See format details below. |
| `--style`         |       | Path    | _None_   | Path to a custom style file (TOML format). See [Style File Format](../file_formats/style.md).                                                   |
| `--annotations`   |       | Path    | _None_   | Path to a custom annotation file (TOML format). See [Annotation File Format](../file_formats/annotations.md).                                   |
| `--dssp`          |       | Path    | _None_   | Path to a DSSP file for secondary structure assignment. **Required if `STRUCTURE_FILE` is in PDB format.** Ignored for mmCIF files.             |
| `--canvas-width`  |       | Integer | `1000`   | Width of the output SVG canvas in pixels.                                                                                                       |
| `--canvas-height` |       | Integer | `1000`   | Height of the output SVG canvas in pixels.                                                                                                      |
| `--show-positions` |       | String  | `minimal` | Position annotation level: `none`, `minimal`, `major`, `full`. Controls residue numbering and terminus labels.                                  |
| `--quiet`         |       | Boolean | `false`  | Suppress all informational output except errors.                                                                                                |
| `--verbose`       |       | Boolean | `false`  | Print additional debug information during execution.                                                                                            |

## Input File Formats

### Structure File (`STRUCTURE_FILE`)

-   Accepts protein structure files in PDB (`.pdb`) or mmCIF (`.cif`, `.mmcif`) format.
-   **PDB Requirement:** If using a `.pdb` file, you **must** provide secondary structure information via a corresponding DSSP file using the `--dssp` option. FlatProt cannot determine secondary structure directly from PDB format.
-   **mmCIF:** Usually contains secondary structure information; the `--dssp` option is typically not needed.

### Matrix File (`--matrix`)

-   Optional. If provided, must be a NumPy file (`.npy`).
-   Specifies a custom 3D transformation (rotation and translation) applied before projection.
-   Expected formats:
    -   A 4x3 matrix: 3x3 rotation matrix in the top 3 rows, 1x3 translation vector in the bottom row.
    -   A 3x4 matrix: Transposed version of the 4x3 format (will be auto-corrected).
    -   A 3x3 matrix: Pure rotation matrix (translation is assumed to be zero).
-   If omitted, the structure is transformed based on its principal axes of inertia.

### Style File (`--style`)

-   Optional. Must be a TOML file.
-   Defines visual properties (colors, strokes, sizes, etc.) for secondary structure elements.
-   See the [Style File Format documentation](../file_formats/style.md) for details.

### Annotation File (`--annotations`)

-   Optional. Must be a TOML file.
-   Defines points, lines, and areas to highlight specific residues or regions.
-   Can include inline styles to customize individual annotations.
-   See the [Annotation File Format documentation](../file_formats/annotations.md) for details.

### DSSP File (`--dssp`)

-   Optional, but **required** when the input `STRUCTURE_FILE` is in PDB format.
-   Standard DSSP output file providing secondary structure assignments (helix, sheet, coil) for each residue.

## Position Annotations (`--show-positions`)

Position annotations add residue numbers and N/C terminus labels to the SVG output, helping identify specific regions of the protein structure. The annotation level controls how much detail is displayed:

### Annotation Levels

-   **`none`**: No position annotations are added to the output.
-   **`minimal`** (default): Only N and C terminus labels are shown, providing basic orientation.
-   **`major`**: N/C terminus labels plus residue numbers for major secondary structures (helices and sheets with ≥3 residues). Short structures are omitted to reduce clutter.
-   **`full`**: All position annotations including residue numbers for all secondary structures, even single-residue elements.

### Position Annotation Details

-   **Terminus Labels**: N and C labels mark the start and end of the protein chain
-   **Residue Numbers**: Start and end residue numbers for secondary structure elements
-   **Filtering**: In 'major' mode, only significant secondary structures (≥3 residues) receive residue numbers
-   **Element Types**: Residue numbers are added only to helices and sheets, not coil regions

## Examples

**Basic Projection (mmCIF to SVG file):**

```bash
flatprot project structure.cif -o output.svg
```

**Projection with DSSP (PDB to SVG file):**

```bash
flatprot project structure.pdb -o output.svg --dssp structure.dssp
```

**Projection to Standard Output (Stdout):**

```bash
flatprot project structure.cif > output.svg
```

**Apply Custom Styling:**

```bash
flatprot project structure.cif -o output.svg --style custom_styles.toml
```

**Add Annotations:**

```bash
flatprot project structure.cif -o output.svg --annotations features.toml
```

**Use a Pre-calculated Alignment Matrix:**

```bash
flatprot project structure.cif -o aligned_output.svg --matrix alignment_matrix.npy
```

**Use a Matrix from PyMOL Orientation:**

```bash
# First, extract matrix from PyMOL
uv run scripts/get_matrix.py structure.cif

# Then use it with project command
flatprot project structure.cif -o pymol_oriented.svg --matrix rotation_matrix.npy
```

See the [PyMOL Matrix Extraction](../tools/matrix_extraction.md) documentation for details.

**Combine Multiple Options:**

```bash
flatprot project structure.cif -o styled_annotated.svg \
    --matrix alignment_matrix.npy \
    --style custom_styles.toml \
    --annotations features.toml \
    --verbose
```

**Adjust Canvas Size:**

```bash
flatprot project structure.cif -o large_output.svg --canvas-width 1500 --canvas-height 1200
```

**Position Annotation Examples:**

```bash
# No position annotations
flatprot project structure.cif -o clean.svg --show-positions none

# Only terminus labels (default)
flatprot project structure.cif -o terminus.svg --show-positions minimal

# Terminus + major secondary structures
flatprot project structure.cif -o major.svg --show-positions major

# All position annotations
flatprot project structure.cif -o detailed.svg --show-positions full
```

## Error Handling

The command provides informative error messages for various issues:

-   **File Errors:** Missing or invalid input files (`FileNotFoundError`, `InvalidStructureError`).
-   **Format Errors:** Incorrect format for structure, matrix, style, or annotation files (`FlatProtError`, specific parser errors).
-   **Missing DSSP:** Error if a PDB file is provided without a `--dssp` file (`FlatProtError`).
-   **Processing Errors:** Issues during transformation, projection, scene creation, or rendering (`FlatProtError`, `CoordinateCalculationError`).
-   **Output Errors:** Failure to write the output SVG file (`OutputFileError`).

Use the `--verbose` flag for more detailed error information and debugging output.
Exit codes are `0` for success and `1` for any error.
