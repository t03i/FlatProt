# Project Command

Transform a 3D protein structure into a standardized 2D SVG representation with optional custom transformations, styles, and annotations.

## Usage

```bash
flatprot project STRUCTURE_FILE [OPTIONS]
```

## Parameters

### Required
- `STRUCTURE_FILE` - Path to the input protein structure file (PDB or mmCIF format)

### Output Options
- `--output` / `-o` - Path to save the output SVG file [default: stdout]
- `--canvas-width` - Width of the output SVG canvas in pixels [default: 1000]
- `--canvas-height` - Height of the output SVG canvas in pixels [default: 1000]

**Note:** Canvas dimensions are controlled via CLI parameters only. Style files do not currently support canvas settings.

### Transformation Options
- `--matrix` - Path to a custom transformation matrix (NumPy `.npy` format). If omitted, inertia-based transformation is applied

### Styling Options
- `--style` - Path to a custom style file (TOML format). See [Style File Format](../file_formats/style.md)
- `--annotations` - Path to a custom annotation file (TOML format). See [Annotation File Format](../file_formats/annotations.md)
- `--show-positions` - Position annotation level: `none`, `minimal`, `major`, `full` [default: minimal]

### Input Options
- `--dssp` - Path to a DSSP file for secondary structure assignment. **Required for PDB format files**

### General Options
- `--quiet` - Suppress all informational output except errors
- `--verbose` - Print additional debug information during execution

## Input File Formats

### Structure Files
- **PDB Format (`.pdb`):** Requires DSSP file for secondary structure assignment
- **mmCIF Format (`.cif`, `.mmcif`):** Usually contains secondary structure information

### Matrix Files
NumPy files (`.npy`) specifying custom 3D transformations. Supported formats:
- 4x3 matrix: 3x3 rotation + 1x3 translation
- 3x4 matrix: Transposed version (auto-corrected)
- 3x3 matrix: Pure rotation (zero translation)

If omitted, inertia-based transformation is applied.

## Position Annotations

Controls residue numbering and terminus labels in the SVG output:

- **`none`**: No position annotations
- **`minimal`** (default): Only N and C terminus labels
- **`major`**: Terminus labels + residue numbers for major secondary structures (≥3 residues)
- **`full`**: All position annotations including single-residue elements

## Examples

### Basic Usage
```bash
# Basic projection (mmCIF to SVG)
flatprot project structure.cif -o output.svg

# Projection with DSSP (PDB to SVG)
flatprot project structure.pdb -o output.svg --dssp structure.dssp

# Output to stdout
flatprot project structure.cif > output.svg
```

### Custom Styling and Annotations
```bash
# Apply custom styling
flatprot project structure.cif -o output.svg --style custom_styles.toml

# Add annotations
flatprot project structure.cif -o output.svg --annotations features.toml

# Combine multiple options
flatprot project structure.cif -o styled_annotated.svg \
    --matrix alignment_matrix.npy \
    --style custom_styles.toml \
    --annotations features.toml
```

### Matrix Transformations
```bash
# Use pre-calculated alignment matrix
flatprot project structure.cif -o aligned_output.svg --matrix alignment_matrix.npy

# Extract matrix from PyMOL orientation
uv run scripts/get_matrix.py structure.cif
flatprot project structure.cif -o pymol_oriented.svg --matrix rotation_matrix.npy
```

See the [PyMOL Matrix Extraction](../tools/matrix_extraction.md) documentation for details.

### Position Annotations
```bash
# No position annotations
flatprot project structure.cif -o clean.svg --show-positions none

# Only terminus labels (default)
flatprot project structure.cif -o terminus.svg --show-positions minimal

# Major secondary structures
flatprot project structure.cif -o major.svg --show-positions major

# All position annotations
flatprot project structure.cif -o detailed.svg --show-positions full
```

### Canvas Size Adjustment
```bash
# Large canvas for detailed output
flatprot project structure.cif -o large_output.svg --canvas-width 1500 --canvas-height 1200
```

## Troubleshooting

### Common Issues

**Missing DSSP file for PDB input:**
```bash
# Generate DSSP file first
mkdssp structure.pdb structure.dssp
flatprot project structure.pdb --dssp structure.dssp -o output.svg
```

**Invalid matrix format:**
- Ensure matrix file is in NumPy `.npy` format
- Supported shapes: 3x3, 3x4, or 4x3
- Use `--verbose` for detailed error information

**File format errors:**
- Verify structure file is valid PDB or mmCIF format
- Check that style/annotation files are valid TOML format
- Ensure all file paths are correct and accessible
