# Overlay Command

Create combined visualizations from multiple protein structures with automatic clustering, family-based alignment, and opacity scaling.

## Overview

The `overlay` command combines multiple protein structures into a single visualization, enabling comparison of structural similarities and differences across protein families. It supports both family-identity alignment (using structure databases) and inertia-based alignment for optimal structure comparison.

## Requirements

For PNG and PDF output, the Cairo graphics library is required:

- **macOS:** `brew install cairo`
- **Ubuntu:** `sudo apt-get install libcairo2-dev pkg-config`
- **Windows:** Install Cairo binaries or use `conda install cairo`

Verify installation:
```bash
python -c "import drawsvg; print('Cairo available:', hasattr(drawsvg, '_cairo_available') and drawsvg._cairo_available)"
```

## Usage

```bash
flatprot overlay FILE_PATTERNS [OPTIONS]
```

## Parameters

### Required
- `FILE_PATTERNS` - Glob pattern(s) or space-separated file paths for input structures
  - Examples: `"structures/*.cif"`, `file1.cif file2.cif`, `"data/*.cif"`

### Output Options
- `--output` - Output file path (format determined by extension: .svg, .png, .pdf) [default: overlay.png]
- `--dpi` - DPI for raster output formats [default: 300]

### Alignment Options
- `--family` - SCOP family ID for fixed family alignment (e.g., "3000114")
- `--alignment-mode` - Alignment strategy [default: family-identity]
  - `family-identity`: Align structures using family database matching
  - `inertia`: Use principal component analysis alignment
- `--min-probability` - Minimum alignment probability threshold [default: 0.5]

### Clustering Options
- `--clustering / --no-clustering` - Enable/disable clustering (auto-enabled at 100+ structures)
- `--clustering-auto-threshold` - Number of structures to auto-enable clustering [default: 100]
- `--clustering-min-seq-id` - Minimum sequence identity for clustering (0.0-1.0) [default: 0.5]
- `--clustering-coverage` - Coverage threshold for clustering (0.0-1.0) [default: 0.9]

### Visualization Options
- `--style` - Custom style TOML file path
- `--canvas-width` - Canvas width in pixels [default: 1000]
- `--canvas-height` - Canvas height in pixels [default: 1000]
- `--disable-scaling` - Disable automatic scaling for consistent size comparisons

### General Options
- `--quiet` - Suppress verbose output
- `--verbose` - Print additional information

## Alignment Modes

### Family-Identity Alignment (Recommended)

Uses FoldSeek to align structures against a curated database of protein families:

- **Automatic Database Download**: Downloads alignment database on first use
- **Batch Processing**: Efficiently processes multiple structures in single FoldSeek call
- **Family-Specific**: Can target specific SCOP family IDs with `--family`
- **Fallback Support**: Automatically falls back to inertia alignment if database alignment fails

```bash
# Align to best matching family
flatprot overlay "structures/*.cif" --alignment-mode family-identity

# Align to specific family
flatprot overlay "toxins/*.cif" --family 3000114 --alignment-mode family-identity
```

### Inertia Alignment

Uses principal component analysis for structure alignment:

- **Fast Processing**: No external database dependencies
- **Universal**: Works with any protein structure
- **Consistent**: Deterministic alignment based on structure geometry

```bash
flatprot overlay "structures/*.cif" --alignment-mode inertia
```

## Clustering

Automatic clustering reduces visual complexity by grouping similar structures:

### Default Behavior
- **Auto-enabled** for datasets with 100+ structures
- Uses FoldSeek to identify structural similarity
- Selects representative structures from each cluster
- Scales opacity based on cluster size (larger clusters = higher opacity)

### Configuration
- **Manual Control**: Use `--clustering` / `--no-clustering` to force enable/disable
- **Auto-threshold**: Adjust when clustering auto-enables with `--clustering-auto-threshold`
- **Similarity Thresholds**: Configure clustering sensitivity:
  - `--clustering-min-seq-id` (0.0-1.0): Lower = more permissive clustering
  - `--clustering-coverage` (0.0-1.0): Lower = more permissive clustering

```bash
# Auto-clustering (default) - enables at 100+ structures
flatprot overlay "small_set/*.cif"  # No clustering (< 100 files)
flatprot overlay "large_dataset/*.cif"  # Auto-clustering (≥ 100 files)

# Manual control
flatprot overlay "structures/*.cif" --clustering  # Force enable
flatprot overlay "structures/*.cif" --no-clustering  # Force disable

# Custom auto-threshold
flatprot overlay "structures/*.cif" --clustering-auto-threshold 50

# Custom clustering parameters
flatprot overlay "structures/*.cif" --clustering-min-seq-id 0.8 --clustering-coverage 0.95
```

## Scaling Options

### Automatic Scaling (Default)
Scales each structure to fit the canvas optimally:
- Good for individual structure visualization
- May cause size inconsistencies between different structures

### Disabled Scaling (New Feature)
Maintains consistent scale across all structures:
- **Use Case**: Comparing structures of different sizes
- **Benefits**: True size relationships preserved
- **Recommendation**: Use with family-identity alignment for best results

```bash
# Compare structures at consistent scale
flatprot overlay "family/*.cif" --disable-scaling --alignment-mode family-identity
```

## Output Formats

### SVG (Vector)
- **Advantages**: Scalable, no Cairo required, small file size
- **Use Cases**: Web display, further editing, publications
- **Limitations**: Limited raster effects

### PNG (Raster)
- **Advantages**: Universal compatibility, predictable appearance
- **Use Cases**: Presentations, papers, thumbnails
- **Requirements**: Cairo graphics library
- **Configuration**: Adjustable DPI (150-600 recommended)

### PDF (Vector)
- **Advantages**: Publication quality, vector graphics, universal
- **Use Cases**: Publications, high-quality prints
- **Requirements**: Cairo graphics library

```bash
# High-quality publication PDF
flatprot overlay "structures/*.cif" -o publication.pdf --dpi 600

# Web-optimized SVG
flatprot overlay "structures/*.cif" -o web_display.svg

# Presentation PNG
flatprot overlay "structures/*.cif" -o presentation.png --dpi 300
```

## Examples

### Basic Usage

**Simple overlay from glob pattern:**
```bash
flatprot overlay "structures/*.cif" -o overlay.png
```

**Multiple file patterns:**
```bash
flatprot overlay file1.cif file2.cif "folder/*.cif" -o combined.svg
```

### Family-Based Alignment

**Automatic family detection:**
```bash
flatprot overlay "3ftx_family/*.cif" --alignment-mode family-identity -o family_aligned.png
```

**Specific family alignment:**
```bash
flatprot overlay "toxins/*.cif" --family 3000114 --alignment-mode family-identity -o toxin_family.pdf
```

**High-confidence alignments only:**
```bash
flatprot overlay "structures/*.cif" --min-probability 0.8 --alignment-mode family-identity
```

### Clustering Configuration

**Strict clustering (more representatives):**
```bash
flatprot overlay "structures/*.cif" --clustering-min-seq-id 0.8 --clustering-coverage 0.95
```

**Permissive clustering (fewer representatives):**
```bash
flatprot overlay "structures/*.cif" --clustering-min-seq-id 0.3 --clustering-coverage 0.7
```

**Force clustering on small datasets:**
```bash
flatprot overlay file1.cif file2.cif file3.cif --clustering
```

**Custom auto-threshold for medium datasets:**
```bash
flatprot overlay "medium_set/*.cif" --clustering-auto-threshold 20
```

### Size Comparison Studies

**Compare structures at true relative sizes:**
```bash
flatprot overlay "different_sizes/*.cif" --disable-scaling --alignment-mode family-identity -o size_comparison.png
```

**Large canvas for detailed comparison:**
```bash
flatprot overlay "complex_structures/*.cif" --canvas-width 1500 --canvas-height 1500 --disable-scaling
```

### Custom Styling

**Apply custom colors and styles:**
```bash
flatprot overlay "structures/*.cif" --style custom_theme.toml -o styled_overlay.png
```

**Publication-ready with high DPI:**
```bash
flatprot overlay "family/*.cif" --style publication.toml --dpi 600 -o figure.pdf
```

### Performance Optimization

**Quick preview (low DPI, small canvas):**
```bash
flatprot overlay "large_set/*.cif" --dpi 150 --canvas-width 500 --canvas-height 500 -o preview.png
```

**Disable clustering for small datasets:**
```bash
flatprot overlay file1.cif file2.cif file3.cif --no-clustering -o small_set.svg
```

## Workflow Details

The overlay command follows this processing pipeline:

1. **Input Resolution**
   - Resolves glob patterns to find structure files
   - Validates file formats and accessibility
   - Reports number of structures found

2. **Optional Clustering**
   - Groups similar structures using FoldSeek (if enabled)
   - Selects representative structures from clusters
   - Calculates opacity weights based on cluster sizes

3. **Alignment Processing**
   - **Family-Identity Mode**:
     - Downloads/validates alignment database
     - Runs batch FoldSeek alignment
     - Retrieves transformation matrices from database
     - Falls back to inertia if alignment fails
   - **Inertia Mode**: Calculates principal component transformations

4. **Structure Projection**
   - Applies transformation matrices to 3D coordinates
   - Projects to 2D using orthographic projection
   - Applies scaling (unless disabled)
   - Centers structures on canvas

5. **Scene Creation**
   - Loads custom styles (if provided)
   - Creates 2D scene objects for each structure
   - Applies opacity based on cluster weights

6. **Overlay Composition**
   - Combines all structure drawings
   - Preserves individual structure opacity
   - Maintains proper layering

7. **Export**
   - Renders to requested format
   - Applies DPI settings for raster formats
   - Saves to specified output path

## Advanced Usage

### Batch Processing Multiple Families

```bash
#!/bin/bash
for family_dir in family_*/; do
    family_name=$(basename "$family_dir")
    flatprot overlay "${family_dir}*.cif" \
        --alignment-mode family-identity \
        --disable-scaling \
        -o "overlays/${family_name}_overlay.pdf" \
        --dpi 300
done
```

### Quality Control Pipeline

```bash
# Generate both aligned and unaligned versions for comparison
flatprot overlay "structures/*.cif" --alignment-mode family-identity -o aligned.png
flatprot overlay "structures/*.cif" --alignment-mode inertia -o inertia.png
flatprot overlay "structures/*.cif" --no-clustering -o all_structures.png
```

### Integration with Analysis Scripts

```python
import subprocess
import json

def create_family_overlay(structures_pattern, family_id=None):
    """Create overlay with optional family targeting."""
    cmd = [
        "flatprot", "overlay", structures_pattern,
        "--alignment-mode", "family-identity",
        "--disable-scaling",
        "-o", f"overlay_{family_id or 'auto'}.png"
    ]

    if family_id:
        cmd.extend(["--family", family_id])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
```

## Performance Tips

### Speed Optimization
- **Use SVG output** for fastest processing (no rasterization)
- **Enable clustering** for large datasets (>10 structures)
- **Lower DPI** for quick previews (150-200)
- **Smaller canvas** for faster processing

### Quality Optimization
- **Higher DPI** for publications (300-600)
- **PDF output** for vector graphics quality
- **Disable clustering** for small, carefully curated sets
- **Custom styles** for publication-ready appearance

### Memory Optimization
- **Process in batches** for very large datasets
- **Use clustering** to reduce memory usage
- **Smaller canvas sizes** for memory-constrained systems

## Troubleshooting

### Common Issues

**"Cairo library not available" error:**
```bash
# Install Cairo
brew install cairo  # macOS
sudo apt-get install libcairo2-dev  # Ubuntu

# Verify installation
python -c "import drawsvg; print(drawsvg._cairo_available)"

# Fallback to SVG
flatprot overlay "*.cif" -o output.svg
```

**"No files found matching pattern" error:**
```bash
# Check pattern syntax
ls structures/*.cif  # Verify files exist

# Use absolute paths
flatprot overlay "/full/path/to/structures/*.cif"

# Quote patterns properly
flatprot overlay "structures/*.cif"  # Correct
flatprot overlay structures/*.cif    # May fail in some shells
```

**Family alignment fails:**
```bash
# Check network connectivity (for database download)
# Verify Foldseek is installed
which foldseek

# Force fallback to inertia mode
flatprot overlay "*.cif" --alignment-mode inertia
```

**Clustering fails:**
```bash
# Disable clustering
flatprot overlay "*.cif" --no-clustering

# Check Foldseek availability
foldseek --help
```

**Memory issues with large datasets:**
```bash
# Use clustering to reduce structures
flatprot overlay "large_set/*.cif" --clustering

# Reduce canvas size
flatprot overlay "*.cif" --canvas-width 500 --canvas-height 500

# Process in smaller batches
flatprot overlay "batch1/*.cif" -o batch1.png
flatprot overlay "batch2/*.cif" -o batch2.png
```

### Performance Benchmarks

Typical processing times on a modern laptop:

| Structures | Mode | Clustering | Time |
|------------|------|------------|------|
| 3-5 | Family-identity | Enabled | 10-30s |
| 3-5 | Inertia | Disabled | 2-5s |
| 10-20 | Family-identity | Enabled | 30-60s |
| 10-20 | Inertia | Enabled | 5-15s |
| 50+ | Family-identity | Enabled | 1-5min |

*First run with family-identity mode includes database download time (~30s)*

## Integration with Other Commands

### Workflow with Align Command

```bash
# 1. First, explore alignment options
flatprot align structure.cif -i alignment_info.json

# 2. Create overlay with discovered family
family_id=$(jq -r '.best_match.family_id' alignment_info.json)
flatprot overlay "family_structures/*.cif" --family "$family_id" -o aligned_overlay.png
```

### Workflow with Project Command

```bash
# 1. Create individual projections
for file in *.cif; do
    flatprot project "$file" -o "individual_${file%.cif}.svg"
done

# 2. Create comparison overlay
flatprot overlay "*.cif" --disable-scaling -o comparison_overlay.png
```

This comprehensive documentation covers all the new features including family-identity alignment, batched processing, disable-scaling option, and provides extensive examples for different use cases.
