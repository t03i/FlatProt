<!--
 Copyright 2025 Tobias Olenyi.
 SPDX-License-Identifier: Apache-2.0
-->

# Installation

FlatProt can be installed using uv (recommended) or pip. The package requires Python 3.11-3.13.

## Quick Installation

### Using uv (Recommended)

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install FlatProt:

```bash
uvx tool install flatprot
```

Or for temporary usage:

```bash
uvx flatprot
```

Verify the installation:

```bash
flatprot --version
```

### Using pip

```bash
pip install flatprot
```

## Development Installation

For development work:

```bash
uv add flatprot
```

## Required Dependencies

FlatProt has several external dependencies that need to be installed separately:

### 1. DSSP (Required for PDB Files)

DSSP is required for secondary structure assignment when working with PDB format files. FlatProt requires **mkdssp version 4.4.0+**.

**Installation Options:**

**macOS (Homebrew):**
```bash
brew install dssp
```

**Ubuntu/Debian:**
```bash
sudo apt-get install dssp
```

**Conda/Mamba:**
```bash
conda install conda-forge::dssp
```

**Manual Installation:**
- Download from: [DSSP Download Page](https://pdb-redo.eu/dssp/download)
- Follow platform-specific build instructions
- Ensure the `mkdssp` executable is in your PATH

**Verification:**
```bash
mkdssp --version
```

### 2. FoldSeek (Required for Alignment Features)

FoldSeek is required for database alignment functionality in the `align` and `split` commands.

**Installation Options:**

**Conda/Mamba (Recommended):**
```bash
conda install bioconda::foldseek
```

**Manual Installation:**
- Download from: [FoldSeek GitHub](https://github.com/steineggerlab/foldseek)
- Follow platform-specific build instructions
- Ensure the `foldseek` executable is in your PATH

**Verification:**
```bash
foldseek --version
```

### 3. Cairo (Optional - Required for PNG/PDF Output)

Cairo is required only for PNG and PDF output from the `flatprot overlay` command. SVG output works without Cairo.

**macOS:**
```bash
brew install cairo
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libcairo2-dev pkg-config
```

**Windows:**
Install Cairo binaries or use conda:
```bash
conda install cairo
```

**Verification:**
```bash
python -c "import drawsvg; print('Cairo available:', hasattr(drawsvg, '_cairo_available') and drawsvg._cairo_available)"
```

## Complete Setup Workflow

Here's a complete setup workflow for different scenarios:

### Basic Usage (CIF files only)
```bash
# Install FlatProt
uvx tool install flatprot

# Test with CIF file (no additional dependencies needed)
flatprot project structure.cif -o output.svg
```

### Full Feature Usage
```bash
# Install FlatProt
uvx tool install flatprot

# Install dependencies
brew install dssp cairo  # macOS
conda install bioconda::foldseek conda-forge::dssp cairo  # Cross-platform

# Test full functionality
flatprot split structure.cif --regions "A:1-100" --show-database-alignment
flatprot overlay "structures/*.cif" -o overlay.png
```


### Platform-Specific Notes

**macOS:**

- Homebrew provides the most reliable installations
- All dependencies available through brew
- On newer M-series models, setting up Cairo requires manual adjustment to find the library path.

**Linux:**

- Use package manager when possible
- Conda/mamba provides consistent cross-distribution support

**Windows:**

- Conda/mamba recommended for all dependencies
- Manual installation may require additional build tools

## Performance Optimization

For better performance, consider:

1. **Use CIF format** when possible
3. **Pre-compute DSSP files** for batch processing of PDB files

## Next Steps

After installation, check out:

- [Getting Started Guide](index.md)
- [Command Line Interface Documentation](commands/)
- [Examples](examples.md)
