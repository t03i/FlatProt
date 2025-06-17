# Split Command

Extract and visualize specific structural regions from protein structures with automatic alignment and progressive gap layout.

## Overview

The `split` command extracts specified structural regions (domains, motifs, binding sites) from protein structures, optionally aligns them using database search, and creates a combined SVG visualization with regions arranged spatially using progressive gap positioning. This enables focused analysis of specific protein regions while maintaining structural context.

## Requirements

- **FoldSeek**: Required for database alignment functionality
  - Install from [FoldSeek GitHub](https://github.com/steineggerlab/foldseek) or use conda: `conda install bioconda::foldseek`
- **DSSP (mkdssp v4.4.0+)**: Required for PDB input files
  - Install from [DSSP GitHub](https://github.com/PDB-ECHA/dssp) or use conda: `conda install conda-forge::dssp`

## Usage

```bash
flatprot split STRUCTURE_FILE --regions "REGIONS" [OPTIONS]
```

## Parameters

### Required
- `STRUCTURE_FILE` - Path to input structure file (PDB/CIF format)
- `--regions` / `-r` - Comma-separated residue regions in format "CHAIN:START-END"
  - Examples: `"A:1-100,A:150-250"`, `"A:1-100,B:50-150,A:200-300"`

### Output Options
- `--output` / `-o` - Output SVG file path [default: split_output.svg]

### Region Gap Options
- `--gap-x` - Progressive horizontal gap between domains in pixels (last domain at origin) [default: 0.0]
- `--gap-y` - Progressive vertical gap between domains in pixels (last domain at origin) [default: 0.0]

### Alignment Options
- `--alignment-mode` - Alignment strategy [default: family-identity]
  - `family-identity`: Align each region using FoldSeek database search with rotation-only transformations
  - `inertia`: Use principal component analysis alignment (no database search)
- `--min-probability` - Minimum alignment probability threshold [default: 0.5]
- `--foldseek` / `-f` - FoldSeek executable path [default: foldseek]
- `--show-database-alignment` - Enable database alignment and show family area annotations [default: False]

### Visualization Options
- `--style` - Custom style TOML file path
- `--canvas-width` - Canvas width in pixels [default: 1000]
- `--canvas-height` - Canvas height in pixels [default: 1000]
- `--show-positions` - Position annotation level: `none`, `minimal`, `major`, `full` [default: minimal]

### Input File Options
- `--dssp` - DSSP file for PDB input (required for PDB files)

## Alignment Modes

### Family-Identity Alignment (Recommended)

Uses FoldSeek to align each region against a curated database of protein families:

- **Automatic Database Download**: Downloads alignment database on first use
- **Region-Specific Alignment**: Each region is aligned independently
- **Rotation-Only Transformations**: Applies only rotation (no translation) to preserve relative positioning
- **Family Annotations**: Shows SCOP family IDs and alignment probabilities when `--show-database-alignment` is enabled

```bash
# Basic family-identity alignment with annotations
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_regions.svg

# High-confidence alignments only
flatprot split protein.cif --regions "A:1-100,A:150-250" --min-probability 0.8 --show-database-alignment
```

### Inertia Alignment

Uses principal component analysis for structure alignment:

- **Fast Processing**: No external database dependencies
- **Universal**: Works with any protein structure
- **Consistent**: Deterministic alignment based on structure geometry

```bash
flatprot split protein.cif --regions "A:1-100,A:150-250" --alignment-mode inertia
```

## Position Annotations (`--show-positions`)

Position annotations add residue numbers and N/C terminus labels to each domain in the split visualization, helping identify specific regions within extracted domains. The annotation level controls the detail displayed for each separated domain:

### Annotation Levels

- **`none`**: No position annotations are added to any domain.
- **`minimal`** (default): Only N and C terminus labels are shown for each domain, providing basic domain orientation.
- **`major`**: N/C terminus labels plus residue numbers for major secondary structures (helices and sheets with ≥3 residues) within each domain. Short structures are omitted to reduce clutter.
- **`full`**: All position annotations including residue numbers for all secondary structures within each domain, even single-residue elements.

### Domain-Specific Behavior

- **Per-Domain Annotations**: Each extracted domain receives its own set of annotations
- **Terminus Labels**: N and C labels mark the start and end of each domain
- **Residue Numbers**: Start and end residue numbers for secondary structure elements within domains
- **Filtering**: In 'major' mode, only significant secondary structures (≥3 residues) receive residue numbers
- **Element Types**: Residue numbers are added only to helices and sheets, not coil regions

## Database Alignment Features

When `--show-database-alignment` is enabled:

### Structural Alignment
- Each region is individually aligned to its best-matching family in the database
- Only rotation transformations are applied (translations are zeroed out)
- Preserves relative positioning while showing aligned orientations
- Regions maintain consistent spatial relationships

### Area Annotations
- SCOP family IDs are displayed as area annotations around each region
- Alignment probabilities are shown as percentages
- Format: "Family ID\n(Probability%)" e.g., "3000622\n(85.3%)"

```bash
# Enable database alignment with area annotations
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o annotated.svg
```

## Progressive Gap Positioning

### Gap Parameters
The split command uses progressive gap positioning where each domain is offset from the previous one:

```bash
# Horizontal gap progression (domains arranged left to right)
flatprot split protein.cif --regions "A:1-100,A:150-250,B:1-80" --gap-x 150

# Vertical gap progression (domains arranged top to bottom)
flatprot split protein.cif --regions "A:1-100,A:150-250,B:1-80" --gap-y 200

# Diagonal arrangement with both horizontal and vertical gaps
flatprot split protein.cif --regions "A:1-100,A:150-250,B:1-80" --gap-x 100 --gap-y 150
```

### Progressive Gap Behavior
- **Last Domain Origin**: The last domain remains at the origin (0,0)
- **Progressive Offset**: Each previous domain is offset by the gap amount multiplied by its position
- **Cumulative Positioning**: Domain 1 at origin, Domain 2 at (-gap_x, -gap_y), Domain 3 at (-2*gap_x, -2*gap_y), etc.
- **Flexible Arrangements**: Combine gap_x and gap_y for custom spatial arrangements

## Region Specification

### Format
- **Pattern**: `"CHAIN:START-END"`
- **Multiple Regions**: Comma-separated list
- **Chain IDs**: Single letters (A, B, C, etc.)
- **Residue Numbers**: 1-based indexing

### Examples

**Single Chain, Multiple Domains:**
```bash
flatprot split protein.cif --regions "A:1-100,A:150-250,A:300-400"
```

**Multiple Chains:**
```bash
flatprot split structure.cif --regions "A:1-100,B:50-150,C:20-120"
```

**Overlapping Regions:**
```bash
flatprot split protein.cif --regions "A:1-120,A:80-200"  # 40 residue overlap
```

## Input File Support

### CIF Format (Recommended)
- Usually contains secondary structure information
- No additional files required
- Better handling of complex structures

```bash
flatprot split structure.cif --regions "A:1-100,A:150-250" -o output.svg
```

### PDB Format
- Requires DSSP file for secondary structure assignment
- Run DSSP separately: `mkdssp -i structure.pdb -o structure.dssp`

```bash
# First, generate DSSP file
mkdssp -i protein.pdb -o protein.dssp

# Then run split command
flatprot split protein.pdb --regions "A:1-100,A:150-250" --dssp protein.dssp -o output.svg
```

## Examples

### Basic Usage

**Simple domain splitting:**
```bash
flatprot split protein.cif --regions "A:1-100,A:150-250" -o domains.svg
```

**Multiple chains with progressive gaps:**
```bash
flatprot split structure.cif --regions "A:1-100,B:50-150,A:200-300" --gap-x 150 --gap-y 100 -o multi_chain.svg
```

### Database Alignment and Annotations

**Enable family alignment with annotations:**
```bash
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_annotated.svg
```

**High-confidence alignments with custom threshold:**
```bash
flatprot split protein.cif --regions "A:1-80,A:100-180,A:200-280" --show-database-alignment --min-probability 0.7 -o high_confidence.svg
```

### Custom Styling and Visualization

**Custom styling with position annotations:**
```bash
# Clean domains without annotations
flatprot split protein.cif --regions "A:10-110,A:130-230" --style custom.toml --show-positions none -o clean.svg

# Basic domain orientation (default)
flatprot split protein.cif --regions "A:10-110,A:130-230" --style custom.toml --show-positions minimal -o oriented.svg

# Major structures with residue numbers
flatprot split protein.cif --regions "A:10-110,A:130-230" --style custom.toml --show-positions major -o detailed.svg

# Full annotation detail
flatprot split protein.cif --regions "A:10-110,A:130-230" --style custom.toml --show-positions full -o comprehensive.svg
```

**Large canvas with progressive gaps:**
```bash
flatprot split protein.cif --regions "A:1-100,A:150-250,A:300-400" --canvas-width 1500 --gap-x 200 --gap-y 100 -o large_canvas.svg
```

### PDB Input Workflow

**Complete PDB processing pipeline:**
```bash
# 1. Generate secondary structure
mkdssp -i protein.pdb -o protein.dssp

# 2. Split with database alignment
flatprot split protein.pdb --regions "A:1-100,A:150-200" --dssp protein.dssp --show-database-alignment -o output.svg
```

### Comparison Studies

**Compare alignment modes:**
```bash
# Family-identity alignment
flatprot split protein.cif --regions "A:1-100,A:150-250" --alignment-mode family-identity --show-database-alignment -o family_aligned.svg

# Inertia alignment for comparison
flatprot split protein.cif --regions "A:1-100,A:150-250" --alignment-mode inertia -o inertia_aligned.svg
```

## Workflow Details

The split command follows this processing pipeline:

1. **Input Validation**
   - Validates structure file format and accessibility
   - Parses and validates region specifications
   - Checks for required DSSP file (PDB input only)

2. **Region Extraction**
   - Extracts specified regions to temporary CIF files
   - Uses GEMMI library for precise region isolation
   - Maintains original residue numbering and chain IDs

3. **Domain-Specific Transformations**
   - **Family-identity mode**: Aligns each region to database using FoldSeek, applies rotation around each domain's center to show recognizable database orientation
   - **Inertia mode**: Calculates individual inertia transformations, applies rotation around each domain's center to highlight secondary structure composition

4. **Centered Rotation and Projection**
   - Rotates each domain around its geometric center, preserving relative positioning
   - Projects 3D coordinates to 2D using orthographic projection
   - Maintains structural relationships while optimizing individual domain orientations

5. **Progressive Gap Application**
   - Applies progressive gap positioning to separate domains spatially
   - Last domain remains at origin, previous domains offset by cumulative gap amounts
   - Preserves relative domain orientations while providing clear separation

6. **Scene Creation**
   - Creates separate groups for each region
   - Applies gap-based positioning transformations
   - Adds area annotations with family IDs and probabilities (if enabled)
   - Applies custom styling (if provided)

7. **SVG Rendering**
   - Combines all region visualizations
   - Maintains proper layering and group organization
   - Exports to SVG format with embedded styling

## Advanced Usage

### Batch Processing Multiple Proteins

```bash
#!/bin/bash
for protein in proteins/*.cif; do
    base=$(basename "$protein" .cif)
    flatprot split "$protein" \
        --regions "A:1-100,A:150-250" \
        --show-database-alignment \
        --gap-x 150 \
        -o "split_results/${base}_split.svg"
done
```

### Domain Family Analysis

```bash
# Extract and analyze common domain across multiple proteins
for protein in family_proteins/*.cif; do
    base=$(basename "$protein" .cif)
    flatprot split "$protein" \
        --regions "A:50-150" \
        --show-database-alignment \
        --min-probability 0.8 \
        -o "domains/${base}_domain.svg"
done
```

### Integration with Analysis Scripts

```python
import subprocess
import json
from pathlib import Path

def split_and_analyze_domains(structure_file, domain_regions, output_dir):
    """Split domains and return alignment information."""
    output_path = output_dir / f"{structure_file.stem}_split.svg"

    cmd = [
        "flatprot", "split", str(structure_file),
        "--regions", ",".join(domain_regions),
        "--show-database-alignment",
        "--gap-x", "150",
        "-o", str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, output_path

# Example usage
domains = ["A:1-100", "A:150-250", "A:300-400"]
success, output = split_and_analyze_domains(
    Path("protein.cif"),
    domains,
    Path("results")
)
```

## Performance Tips

### Speed Optimization

- **Use CIF input** (no DSSP file generation required)
- **Disable database alignment** for quick gap-based positioning only
- **Smaller canvas sizes** for faster processing
- **Fewer regions** reduce extraction and alignment time

### Quality Optimization

- **Enable database alignment** for biologically meaningful orientations
- **Higher alignment probability thresholds** for confident annotations
- **Custom styling** for publication-ready output
- **Larger canvas sizes** for detailed visualization

### Memory Optimization

- **Limit number of regions** for memory-constrained systems
- **Smaller region sizes** reduce memory usage
- **Use minimal/none position annotations** for simpler scenes (use `--show-positions none` for cleanest output)

## Troubleshooting

### Common Issues

**"DSSP file required for PDB input" error:**

```bash
# Generate DSSP file
mkdssp -i structure.pdb -o structure.dssp

# Then retry with DSSP file
flatprot split structure.pdb --regions "A:1-100" --dssp structure.dssp
```

**"Invalid region format" error:**

```bash
# Check region format
flatprot split protein.cif --regions "A:1-100,B:50-150"  # Correct
flatprot split protein.cif --regions "A:1:100,B:50-150"  # Incorrect (colon instead of dash)
```

**"No successful alignments found" error:**

```bash
# Lower probability threshold
flatprot split protein.cif --regions "A:1-100" --min-probability 0.3 --show-database-alignment

# Or disable database alignment
flatprot split protein.cif --regions "A:1-100" --alignment-mode inertia
```

**"Chain not found in structure" error:**

```bash
# Check available chains
grep "^ATOM" structure.pdb | awk '{print $5}' | sort -u  # For PDB
# Or examine CIF structure to verify chain IDs
```

**Database download issues:**

```bash
# Check network connectivity
# Verify FoldSeek installation
which foldseek

# Use inertia mode as fallback
flatprot split protein.cif --regions "A:1-100" --alignment-mode inertia
```

### Performance Benchmarks

Typical processing times on a modern laptop:
*First run with family-identity mode includes database download time (~30s)*

| Regions | Mode | Database Alignment | Time |
|---------|------|-------------------|------|
| 2-3 | Inertia | Disabled | 2-5s |
| 2-3 | Family-Identity | Enabled | 15-30s |
| 5-7 | Inertia | Disabled | 3-8s |
| 5-7 | Family-Identity | Enabled | 30-60s |


## Integration with Other Commands

### Workflow with Align Command

```bash
# 1. First, explore protein family alignment
flatprot align protein.cif -i alignment_info.json

# 2. Extract domains with family-specific alignment
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_domains.svg
```

### Workflow with Project Command

```bash
# 1. Create full structure projection
flatprot project protein.cif -o full_structure.svg

# 2. Create domain-specific split view with progressive gaps
flatprot split protein.cif --regions "A:1-100,A:150-250" --gap-x 150 -o domain_split.svg
```
