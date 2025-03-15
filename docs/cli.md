# FlatProt CLI Documentation

FlatProt provides a command-line interface (CLI) for generating 2D protein visualizations from structure files. The CLI allows you to transform protein structures into standardized 2D representations, apply custom styles, add annotations, and save the results as SVG files.

## Installation

To use the FlatProt CLI, you need to install the FlatProt package:

```bash
pip install FlatProt
```

## Basic Usage

The basic syntax for the FlatProt CLI is:

```bash
flatprot STRUCTURE_FILE [OUTPUT_FILE] [OPTIONS]
```

Where:

-   `STRUCTURE_FILE` is the path to a PDB or CIF structure file (required)
-   `OUTPUT_FILE` is the path for the output SVG file (optional, defaults to stdout)

### Example

```bash
flatprot structure.cif output.svg
```

## Command Options

| Option               | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| `--matrix PATH`      | Path to a custom transformation matrix file (NumPy format) |
| `--style PATH`       | Path to a TOML file with custom style definitions          |
| `--annotations PATH` | Path to a TOML file with annotation definitions            |
| `--dssp PATH`        | Path to a DSSP file with secondary structure assignments   |
| `--quiet`            | Suppress all output except errors                          |
| `--verbose`          | Print additional information                               |

## Input File Formats

### Structure Files

FlatProt accepts protein structure files in the following formats:

-   PDB (.pdb)
-   mmCIF (.cif, .mmcif)

Note: When using PDB files, you must also provide a DSSP file using the `--dssp` option, as secondary structure information cannot be extracted directly from PDB files. **This is not required if using a CIF file enriched with secondary structure information.**

### Matrix Files

Custom transformation matrices should be provided as NumPy (.npy) files containing a 3x3 or 4x4 transformation matrix. If not provided, FlatProt uses an inertia-based transformation by default.

### Style Files

Style files use the TOML format to define visual properties for different elements of the visualization. Here's an example of a style file:

```toml
[helix]
fill_color = "#FF5733"
stroke_color = "#000000"
amplitude = 0.5

[sheet]
fill_color = "#33FF57"
line_width = 2.0
min_sheet_length = 3

[point]
fill_color = "#3357FF"
radius = 5.0

[line]
stroke_color = "#FF33A8"
stroke_width = 2.0

[area]
fill_color = "#FFFF33"
fill_opacity = 0.5
stroke_color = "#000000"
padding = 2.0
smoothing_window = 3
interpolation_points = 10
```

### Annotation Files

Annotation files use the TOML format to define points, lines, and areas to highlight in the visualization. Here's an example:

```toml
[[annotations]]
type = "point"
label = "Active Site"
indices = [45]
chain = "A"

[[annotations]]
type = "line"
label = "Disulfide Bond"
indices = [23, 76]
chain = "A"

[[annotations]]
type = "area"
label = "Binding Domain"
range = { start = 100, end = 150 }
chain = "A"
```

## Examples

### Basic Visualization

Generate a basic 2D visualization of a protein structure:

```bash
flatprot structure.cif output.svg
```

### Custom Styling

Apply custom styles to the visualization:

```bash
flatprot structure.cif output.svg --style styles.toml
```

### Adding Annotations

Highlight specific features in the visualization:

```bash
flatprot structure.cif output.svg --annotations annotations.toml
```

### Custom Transformation

Use a custom transformation matrix:

```bash
flatprot structure.cif output.svg --matrix custom_matrix.npy
```

### Using DSSP for Secondary Structure

Provide secondary structure information for a PDB file:

```bash
flatprot structure.pdb output.svg --dssp structure.dssp
```

### Combining Options

You can combine multiple options:

```bash
flatprot structure.cif output.svg --style styles.toml --annotations annotations.toml --verbose
```

## Error Handling

The CLI provides informative error messages for common issues:

-   Missing or invalid structure files
-   Incompatible file formats
-   Missing required secondary structure information
-   Invalid annotation or style specifications

Use the `--verbose` flag to get more detailed information when errors occur.
