# Overlay Command

Create combined visualizations from multiple protein structures with automatic clustering and alignment.

## Requirements

PNG and PDF output require the Cairo graphics library:

- **macOS:** `brew install cairo`
- **Ubuntu:** `sudo apt-get install libcairo2-dev pkg-config`
- **Windows:** Install Cairo binaries or use `conda install cairo`

Verify installation: `python -c "import drawsvg; print('Cairo available:', hasattr(drawsvg, '_cairo_available') and drawsvg._cairo_available)"`

## Usage

```bash
flatprot overlay INPUT_PATTERN [OPTIONS]
```

## Parameters

### Required
- `input_pattern` - Glob pattern or file paths for input structures (e.g., `"structures/*.cif"`,)

### Output Options
- `--output`, `-o` - Output file path (format determined by extension: .svg, .png, .pdf) [default: overlay.png]
- `--dpi` - DPI for raster output formats [default: 300]

### Alignment Options
- `--family` - SCOP family ID for fixed family alignment (e.g., "3000114")
- `--alignment-mode` - Alignment strategy: "family-identity" or "inertia" [default: family-identity]
- `--min-probability` - Minimum alignment probability threshold [default: 0.5]

### Visualization Options
- `--style` - Custom style TOML file path
- `--canvas-width` - Canvas width in pixels [default: 1000]
- `--canvas-height` - Canvas height in pixels [default: 1000]
- `--no-clustering` - Disable automatic structure clustering

### General Options
- `--quiet` - Suppress verbose output

## Examples

### Basic Overlay from Glob Pattern
```bash
flatprot overlay "structures/*.cif" -o overlay.png
```

### Align to Specific Family with High DPI
```bash
flatprot overlay *.cif --family 3000114 -o result.pdf --dpi 600
```

### Use Inertia Alignment Without Clustering
```bash
flatprot overlay file1.cif file2.cif --alignment-mode inertia --no-clustering
```

### Custom Styling
```bash
flatprot overlay "data/*.cif" --style custom.toml -o styled_overlay.png
```

### Large Canvas for Complex Overlays
```bash
flatprot overlay "proteins/*.cif" --canvas-width 1500 --canvas-height 1500 -o large.png
```

## Workflow

1. **Input Resolution:** Resolves glob patterns to find structure files
2. **Optional Clustering:** Groups similar structures using Foldseek (reduces visual clutter)
3. **Alignment:** Aligns structures using specified mode:
   - `family-identity`: Aligns to specific SCOP family or database search
   - `inertia`: Uses principal component analysis alignment
4. **Projection:** Generates 2D projections for each aligned structure
5. **Overlay Creation:** Combines projections with opacity scaling based on cluster size
6. **Export:** Saves in requested format (SVG/PNG/PDF)

## Clustering

When enabled (default), structures are automatically clustered using Foldseek:
- Groups similar structures together
- Selects representative structures from each cluster
- Scales opacity based on cluster size (larger clusters = higher opacity)
- Reduces visual complexity for large datasets

Disable with `--no-clustering` to include all structures.

## Output Formats

- **SVG:** Vector format, no Cairo required, scalable
- **PNG:** Raster format, requires Cairo, good for papers/presentations
- **PDF:** Vector format, requires Cairo, publication quality

## Performance Tips

- Use clustering for datasets with >5 structures
- Consider lower DPI (150-200) for quick previews
- Use smaller canvas sizes for faster processing
- SVG output is fastest (no rasterization)

## Troubleshooting

**"Cairo library not available" error:**
- Install Cairo as described in Requirements section
- Verify installation with the provided command
- Use SVG output as fallback

**"No files found matching pattern" error:**
- Check glob pattern syntax and file paths
- Ensure files have .cif extension
- Use absolute paths if relative paths fail

**Clustering fails:**
- Command falls back to using all structures
- Check that Foldseek is installed and in PATH
- Disable clustering with `--no-clustering`
