# Split Command

Extract and visualize specific structural regions from protein structures with automatic alignment and progressive gap layout.

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
- `--canvas-width` - Canvas width in pixels [default: 1000]
- `--canvas-height` - Canvas height in pixels [default: 1000]

### Gap Options
- `--gap-x` - Progressive horizontal gap between domains in pixels [default: 0.0]
- `--gap-y` - Progressive vertical gap between domains in pixels [default: 0.0]

### Alignment Options
- `--alignment-mode` - Alignment strategy [default: family-identity]
  - `family-identity`: Align each region using FoldSeek database search
  - `inertia`: Use principal component analysis alignment
- `--min-probability` - Minimum alignment probability threshold [default: 0.5]
- `--foldseek` / `-f` - FoldSeek executable path [default: foldseek]
- `--show-database-alignment` - Enable database alignment and show family area annotations

### Styling Options
- `--style` - Custom style TOML file path
- `--show-positions` - Position annotation level: `none`, `minimal`, `major`, `full` [default: minimal]

### Input Options
- `--dssp` - DSSP file for PDB input (required for PDB files)

## Requirements

- **FoldSeek**: Required for database alignment functionality
- **DSSP (mkdssp v4.4.0+)**: Required for PDB input files

Install via conda:
```bash
conda install bioconda::foldseek conda-forge::dssp
```

## Alignment Modes

### Family-Identity Alignment (Recommended)
Uses FoldSeek to align each region against a curated database of protein families:
- Automatic database download on first use
- Region-specific alignment with rotation-only transformations
- Family annotations show SCOP family IDs and alignment probabilities

```bash
# Basic family-identity alignment with annotations
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_regions.svg

# High-confidence alignments only
flatprot split protein.cif --regions "A:1-100,A:150-250" --min-probability 0.8 --show-database-alignment
```

### Inertia Alignment
Uses principal component analysis for structure alignment. Fast processing with no external database dependencies.

```bash
flatprot split protein.cif --regions "A:1-100,A:150-250" --alignment-mode inertia
```

## Position Annotations

Controls residue numbering and terminus labels for each domain:

- **`none`**: No position annotations
- **`minimal`** (default): Only N and C terminus labels for each domain
- **`major`**: Terminus labels + residue numbers for major secondary structures (≥3 residues)
- **`full`**: All position annotations including single-residue elements

## Progressive Gap Positioning

Each domain is offset from the previous one using progressive gaps:
- **Last domain** remains at origin (0,0)
- **Previous domains** offset by gap amount × position
- **Flexible arrangements** combine gap_x and gap_y for custom layouts

```bash
# Horizontal arrangement
flatprot split protein.cif --regions "A:1-100,A:150-250,B:1-80" --gap-x 150

# Vertical arrangement
flatprot split protein.cif --regions "A:1-100,A:150-250,B:1-80" --gap-y 200

# Diagonal arrangement
flatprot split protein.cif --regions "A:1-100,A:150-250,B:1-80" --gap-x 100 --gap-y 150
```

## Region Specification

### Format
- **Pattern**: `"CHAIN:START-END"`
- **Multiple regions**: Comma-separated list
- **Chain IDs**: Single letters (A, B, C, etc.)
- **Residue numbers**: 1-based indexing

### Examples
```bash
# Single chain, multiple domains
flatprot split protein.cif --regions "A:1-100,A:150-250,A:300-400"

# Multiple chains
flatprot split structure.cif --regions "A:1-100,B:50-150,C:20-120"

# Overlapping regions
flatprot split protein.cif --regions "A:1-120,A:80-200"  # 40 residue overlap
```

## Examples

### Basic Usage
```bash
# Simple domain splitting
flatprot split protein.cif --regions "A:1-100,A:150-250" -o domains.svg

# Multiple chains with progressive gaps
flatprot split structure.cif --regions "A:1-100,B:50-150,A:200-300" --gap-x 150 --gap-y 100 -o multi_chain.svg
```

### Database Alignment and Annotations
```bash
# Enable family alignment with annotations
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_annotated.svg

# High-confidence alignments with custom threshold
flatprot split protein.cif --regions "A:1-80,A:100-180,A:200-280" --show-database-alignment --min-probability 0.7 -o high_confidence.svg
```

### Custom Styling and Visualization
```bash
# Clean domains without annotations
flatprot split protein.cif --regions "A:10-110,A:130-230" --style custom.toml --show-positions none -o clean.svg

# Major structures with residue numbers
flatprot split protein.cif --regions "A:10-110,A:130-230" --style custom.toml --show-positions major -o detailed.svg

# Large canvas with progressive gaps
flatprot split protein.cif --regions "A:1-100,A:150-250,A:300-400" --canvas-width 1500 --gap-x 200 --gap-y 100 -o large_canvas.svg
```

### PDB Input Workflow
```bash
# 1. Generate secondary structure
mkdssp -i protein.pdb -o protein.dssp

# 2. Split with database alignment
flatprot split protein.pdb --regions "A:1-100,A:150-200" --dssp protein.dssp --show-database-alignment -o output.svg
```

### Comparison Studies
```bash
# Compare alignment modes
flatprot split protein.cif --regions "A:1-100,A:150-250" --alignment-mode family-identity --show-database-alignment -o family_aligned.svg
flatprot split protein.cif --regions "A:1-100,A:150-250" --alignment-mode inertia -o inertia_aligned.svg
```

## Troubleshooting

### Common Issues

**"DSSP file required for PDB input" error:**
```bash
# Generate DSSP file first
mkdssp -i structure.pdb -o structure.dssp
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
# Check available chains in PDB
grep "^ATOM" structure.pdb | awk '{print $5}' | sort -u
# Or examine CIF structure to verify chain IDs
```

**Database download issues:**
```bash
# Check network connectivity and FoldSeek installation
which foldseek

# Use inertia mode as fallback
flatprot split protein.cif --regions "A:1-100" --alignment-mode inertia
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
- **Use minimal/none position annotations** for simpler scenes

## Integration with Other Commands

### Workflow with Align Command
```bash
# 1. Explore protein family alignment
flatprot align protein.cif -i alignment_info.json

# 2. Extract domains with family-specific alignment
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_domains.svg
```

### Workflow with Project Command
```bash
# 1. Create full structure projection
flatprot project protein.cif -o full_structure.svg

# 2. Create domain-specific split view
flatprot split protein.cif --regions "A:1-100,A:150-250" --gap-x 150 -o domain_split.svg
```
