# Overlay Command

Create combined visualizations from multiple protein structures with automatic clustering, family-based alignment, and opacity scaling.

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
Uses FoldSeek to align structures against a curated database of protein families with automatic database download and batch processing.

**For optimal conservation**, it is recommended to align all structures to the same common core structure using a fixed family ID. This ensures consistent alignment across all structures in the overlay.

```bash
# Align to best matching family (may vary per structure)
flatprot overlay "structures/*.cif" --alignment-mode family-identity

# Align to specific family for optimal conservation (recommended)
flatprot overlay "toxins/*.cif" --family 3000114 --alignment-mode family-identity
```

**Note:** Using automatic family detection or other alignment modes does not guarantee optimal conservation across structures, as each structure may align to different reference families.

### Inertia Alignment
Uses principal component analysis for structure alignment. Fast processing with no external database dependencies.

```bash
flatprot overlay "structures/*.cif" --alignment-mode inertia
```

## Clustering

Automatic clustering reduces visual complexity by grouping similar structures:

- **Auto-enabled** for datasets with 100+ structures
- Uses FoldSeek to identify structural similarity
- Scales opacity based on cluster size (larger clusters = higher opacity)

```bash
# Manual control
flatprot overlay "structures/*.cif" --clustering  # Force enable
flatprot overlay "structures/*.cif" --no-clustering  # Force disable

# Custom thresholds
flatprot overlay "structures/*.cif" --clustering-min-seq-id 0.8 --clustering-coverage 0.95
```

## Output Formats

### SVG (Vector)
- Scalable, no Cairo required, small file size
- Best for web display and publications

### PNG (Raster)
- Universal compatibility, requires Cairo graphics library
- Adjustable DPI (150-600 recommended)

### PDF (Vector)
- Publication quality, requires Cairo graphics library

```bash
# High-quality publication PDF
flatprot overlay "structures/*.cif" -o publication.pdf --dpi 600

# Web-optimized SVG
flatprot overlay "structures/*.cif" -o web_display.svg
```

## Requirements

For PNG and PDF output, install Cairo graphics library:

- **macOS:** `brew install cairo`
- **Ubuntu:** `sudo apt-get install libcairo2-dev pkg-config`
- **Windows:** `conda install cairo`

Verify installation:
```bash
python -c "import drawsvg; print('Cairo available:', hasattr(drawsvg, '_cairo_available') and drawsvg._cairo_available)"
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

### Size Comparison Studies

**Compare structures at true relative sizes:**
```bash
flatprot overlay "different_sizes/*.cif" --disable-scaling --alignment-mode family-identity -o size_comparison.png
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

## Troubleshooting

### Common Issues

**"Cairo library not available" error:**
```bash
# Install Cairo
brew install cairo  # macOS
sudo apt-get install libcairo2-dev  # Ubuntu

# Fallback to SVG
flatprot overlay "*.cif" -o output.svg
```

**"No files found matching pattern" error:**
```bash
# Quote patterns properly
flatprot overlay "structures/*.cif"  # Correct
```

**Family alignment fails:**
```bash
# Force fallback to inertia mode
flatprot overlay "*.cif" --alignment-mode inertia
```

**Memory issues with large datasets:**
```bash
# Use clustering to reduce structures
flatprot overlay "large_set/*.cif" --clustering

# Reduce canvas size
flatprot overlay "*.cif" --canvas-width 500 --canvas-height 500
```

## Performance Tips

- **Use SVG output** for fastest processing (no rasterization)
- **Enable clustering** for large datasets (>10 structures)
- **Lower DPI** for quick previews (150-200)
- **Higher DPI** for publications (300-600)
