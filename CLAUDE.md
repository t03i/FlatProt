# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlatProt is a Python package for creating simplified 2D protein visualizations, focusing on comparable representations for same-family proteins. The main CLI commands are:

- `flatprot project` - Creates 2D SVG projections from protein structures
- `flatprot align` - Aligns protein structures using rotation
- `flatprot overlay` - Creates overlay visualizations from multiple protein structures

## Development Commands

### Package Management
- Install dependencies: `uv sync`
- Install with dev dependencies: `uv sync --group dev`
- Install with database builder dependencies: `uv sync --group db-builder`

### Testing
- Only run python or pytest through uv
- Run all tests: `pytest`
- Run specific test module: `pytest tests/core/test_structure.py`
- Run tests with verbose output: `pytest -v`

### Building and Installation
- Build package: `uv build`
- Install locally: `uv tool install .`

### Documentation
- Build docs: `mkdocs build`
- Serve docs locally: `mkdocs serve`

## Architecture Overview

### Core Components

**CLI Layer** (`src/flatprot/cli/`):
- `main.py` - Main CLI app using cyclopts framework
- `project.py` - Implements `flatprot project` command for structure projection
- `align.py` - Implements `flatprot align` command for structure alignment
- `overlay.py` - Implements `flatprot overlay` command for multi-structure overlays

**Core Layer** (`src/flatprot/core/`):
- `structure.py` - Core protein structure representation
- `coordinates.py` - 3D coordinate handling and transformations
- `types.py` - Type definitions used throughout the package

**I/O Layer** (`src/flatprot/io/`):
- `structure.py` - Structure file parsing (PDB, CIF)
- `structure_gemmi_adapter.py` - GEMMI library integration for structure parsing
- `dssp.py` - DSSP secondary structure parsing
- `annotations.py` - Annotation file handling (TOML format)
- `styles.py` - Style configuration parsing
- `matrix.py` - Matrix data I/O

**Scene System** (`src/flatprot/scene/`):
- `scene.py` - Main scene container for 2D visualization elements
- `structure/` - 2D structure elements (helices, sheets, coils)
- `annotation/` - 2D annotation elements (points, lines, areas)
- `resolver.py` - Resolves coordinate conflicts and overlaps

**Projection** (`src/flatprot/projection/`):
- `orthographic.py` - 3D to 2D orthographic projection
- `base.py` - Base projection interface

**Rendering** (`src/flatprot/renderers/`):
- `svg_renderer.py` - Main SVG output renderer
- `svg_structure.py` - SVG structure element rendering
- `svg_annotations.py` - SVG annotation rendering

**Transformation** (`src/flatprot/transformation/`):
- `inertia_transformation.py` - Principal component analysis for structure alignment
- `matrix_transformation.py` - Matrix-based transformations

**Alignment** (`src/flatprot/alignment/`):
- `foldseek.py` - Foldseek integration for structure alignment
- `db.py` - Database operations for alignment

**Utilities** (`src/flatprot/utils/`):
- `overlay_utils.py` - Multi-structure overlay creation and clustering utilities
- `structure_utils.py` - Structure transformation and projection utilities
- `scene_utils.py` - Scene creation and annotation utilities

### Data Flow

1. **Input Processing**: Structure files (PDB/CIF) are parsed via GEMMI adapter
2. **Transformation**: 3D coordinates undergo PCA-based alignment and rotation
3. **Projection**: 3D coordinates are projected to 2D using orthographic projection
4. **Scene Assembly**: 2D elements (structures + annotations) are assembled in scene
5. **Conflict Resolution**: Overlapping elements are resolved via scene resolver
6. **Rendering**: Final SVG is generated with styled elements

### Key Dependencies

- **GEMMI**: Structure file parsing and manipulation
- **Polars**: Data processing and analysis
- **Pydantic**: Data validation and settings management
- **DrawSVG**: SVG generation and rendering
- **Cyclopts**: CLI framework
- **NumPy**: Numerical operations

### External Tools Integration

- **Foldseek**: Structure alignment tool (path required as CLI argument)
- **DSSP**: Secondary structure assignment (mkdssp version 4.4.0 required)

### File Formats

- **Input**: PDB, CIF structure files (require headers for predicted structures)
- **Annotations**: TOML format for custom annotations
- **Styles**: TOML format for visualization styling
- **Output**: SVG, PNG, PDF formats for 2D visualizations (PNG/PDF require Cairo)

### Testing Structure

Tests are organized by component:
- `tests/core/` - Core functionality tests
- `tests/io/` - I/O operation tests
- `tests/scene/` - Scene system tests
- `tests/integration/` - End-to-end workflow tests
- `tests/utils/` - Utility function tests
- `tests/cli/` - CLI command tests

## CLI Commands Reference

### flatprot project
Creates 2D SVG projections from protein structures.

**Usage**: `flatprot project STRUCTURE_FILE [OPTIONS]`

**Key Options**:
- `--output/-o` - SVG output path (defaults to stdout)
- `--matrix` - Custom transformation matrix (.npy format)
- `--style` - Custom style file (TOML format)
- `--annotations` - Annotation file (TOML format)
- `--dssp` - DSSP file (required for PDB input)
- `--canvas-width/--canvas-height` - Canvas dimensions (default 1000x1000)

**Examples**:
```bash
# Basic CIF projection
flatprot project structure.cif -o output.svg

# PDB with DSSP
flatprot project structure.pdb --dssp structure.dssp -o output.svg

# With custom styling and annotations
flatprot project structure.cif -o output.svg --style custom.toml --annotations features.toml
```

### flatprot align
Finds best matching protein superfamily using Foldseek and retrieves transformation matrix.

**Usage**: `flatprot align STRUCTURE_FILE [OPTIONS]`

**Key Options**:
- `--matrix/-m` - Output matrix path (default: alignment_matrix.npy)
- `--info/-i` - Alignment info JSON output
- `--foldseek/-f` - Foldseek executable path
- `--foldseek-db/-b` - Custom Foldseek database
- `--min-probability/-p` - Alignment probability threshold (default 0.5)
- `--alignment-mode/-a` - `family-identity` (default) or `family-inertia`

**Examples**:
```bash
# Basic alignment
flatprot align protein.pdb

# With custom threshold and output paths
flatprot align protein.cif -m rotation.npy -i info.json --min-probability 0.7
```

### flatprot overlay
Creates overlay visualizations from multiple protein structures with opacity scaling based on clustering.

**Usage**: `flatprot overlay FILE_PATTERNS [OPTIONS]`

**Key Options**:
- `--output/-o` - Output file path (default: overlay.png)
- `--family` - SCOP family ID for fixed family alignment (e.g., "3000114")
- `--alignment-mode` - Alignment strategy: `family-identity` (default) or `inertia`
- `--style` - Custom style file (TOML format)
- `--canvas-width/--canvas-height` - Canvas dimensions (default 1000x1000)
- `--min-probability` - Alignment probability threshold (default 0.5)
- `--dpi` - DPI for raster output formats (default 300)
- `--no-clustering` - Disable automatic structure clustering

**Output Formats**:
- SVG (no Cairo required)
- PNG/PDF (requires Cairo: `brew install cairo` on macOS)

**Examples**:
```bash
# Basic overlay from glob pattern
flatprot overlay "structures/*.cif" -o overlay.png

# Multiple file patterns
flatprot overlay file1.cif file2.cif folder/*.cif -o overlay.pdf

# With family alignment and custom settings
flatprot overlay "data/*.cif" --family 3000114 --dpi 600 -o high_res.png

# Inertia-based alignment without clustering
flatprot overlay "*.cif" --alignment-mode inertia --no-clustering -o simple.svg
```

## File Formats

### Structure Files
- **Input**: PDB (.pdb) or mmCIF (.cif, .mmcif)
- **PDB Requirements**: Must provide DSSP file via `--dssp` option
- **mmCIF**: Usually contains secondary structure information
- **Headers Required**: Input PDB files need headers (important for predicted structures)

### Matrix Files (.npy)
NumPy arrays defining 3D transformations:
- **4×3 matrix**: 3×3 rotation + 1×3 translation (preferred)
- **3×4 matrix**: Transposed version (auto-corrected)
- **3×3 matrix**: Rotation only (zero translation)

### Style Files (.toml)
Customize visualization appearance:
- `[canvas]` - Canvas properties (width, height, background, margin)
- `[helix]` - Alpha helix styling (color, thickness, wavelength, amplitude)
- `[sheet]` - Beta sheet styling (color, arrow width, min length)
- `[coil]` - Coil regions (color, stroke, smoothing)
- `[point_annotation]`, `[line_annotation]`, `[area_annotation]` - Annotation styles

### Annotation Files (.toml)
Define highlighting features:
- **Point annotations**: `type = "point"`, `index = "A:41"`
- **Line annotations**: `type = "line"`, `indices = ["A:23", "A:76"]`
- **Area annotations**: `type = "area"`, `range = "A:100-150"`
- **Inline styles**: Override defaults per annotation

## Workflow Patterns

### Standard 2D Projection
1. Obtain structure file (PDB/CIF)
2. Calculate secondary structure: `mkdssp -i structure.cif -o structure.dssp`
3. Generate projection: `flatprot project structure.cif -o projection.svg`

### Family-Based Alignment
1. Align structure: `flatprot align protein.pdb -m matrix.npy`
2. Project with alignment: `flatprot project protein.pdb --matrix matrix.npy -o aligned.svg`

### Custom Visualization
1. Create style file defining colors, dimensions
2. Create annotation file highlighting features
3. Apply both: `flatprot project structure.cif --style custom.toml --annotations features.toml -o styled.svg`

### Multi-Structure Overlay
1. Collect related structures: `ls structures/*.cif`
2. Create overlay: `flatprot overlay "structures/*.cif" -o overlay.png`
3. Optional: Add family alignment: `flatprot overlay "*.cif" --family 3000114 -o aligned_overlay.pdf`

## Visualization Types
- **Normal**: Standard detailed secondary structure representation
- **Simple-coil**: Simplified coil representation
- **Only-path**: Minimalist path representation
- **Special**: Enhanced with customizable features

## Secondary Structure Colors (Default)
- **Helices**: Red (#F05039)
- **Sheets**: Blue (#1F449C)
- **Coils**: Black with adjustable simplification

## Implemented Use Cases and Data Flow Patterns

Based on the examples folder, FlatProt supports several key use cases with distinct data flow patterns:

### 1. Family Alignment and Comparison (`3ftx_alignment.py`)
**Use Case**: Align related protein structures (same family) for comparative visualization

**Data Flow**:
```
Structure Files (CIF) → flatprot align → Transformation Matrices → flatprot project → Aligned SVGs
```

**Key Steps**:
1. Multiple related structures (cobra.cif, krait.cif, snake.cif)
2. Each structure aligned against reference database using `flatprot align --min-probability 0.5`
3. Alignment produces transformation matrices (.npy) and info files (.json)
4. Structures projected using matrices: `flatprot project --matrix alignment_matrix.npy`
5. Result: Comparable 2D projections with consistent orientation

**Pattern**: Batch processing of related structures for comparative analysis

### 2. Custom Styling and Annotations (`annotations.py`)
**Use Case**: Apply custom visual styles and highlight specific features

**Data Flow**:
```
Structure File → Custom Style/Annotation Files (TOML) → flatprot project → Styled SVG
```

**Key Steps**:
1. Define custom styles in TOML (helix/sheet/coil colors, opacity)
2. Create annotation file with point/line/area annotations
3. Project with custom files: `flatprot project --style custom.toml --annotations features.toml`

**Pattern**: Visual customization and feature highlighting

### 3. Domain-Based Analysis (`chainsaw.py`)
**Use Case**: Domain-aware visualization and alignment

**Data Flow**:
```
Structure + Domain Definitions → Domain Extraction → Individual Alignment → Domain Transformation → Multi-Modal Projections
```

**Key Steps**:
1. Parse domain definitions from Chainsaw TSV files
2. Extract individual domains using GEMMI library
3. Align each domain separately using `flatprot align`
4. Apply domain transformations to original structure
5. Generate three projection types:
   - Normal projection (standard inertia-based)
   - Domain-aligned projection (domains individually aligned then reassembled)
   - Domain-separated projection (domains spatially separated)

**Pattern**: Complex multi-step processing with domain-specific operations

### 4. Feature-Specific Annotation (`dysulfide.py`)
**Use Case**: Compute and visualize structural features (disulfide bonds)

**Data Flow**:
```
Structure Files → Feature Computation (GEMMI) → Annotation Generation → Alignment → Projection
```

**Key Steps**:
1. Batch process multiple structures from directory
2. Compute cystine bridges using distance thresholds (GEMMI library)
3. Generate annotation TOML files for identified bridges
4. Align structures: `flatprot align --min-probability 0.5`
5. Project with annotations: `flatprot project --annotations bridges.toml --matrix alignment.npy`

**Pattern**: Automated feature detection and visualization

### 5. Multi-Structure Overlay (Built-in `flatprot overlay`)
**Use Case**: Create overlay visualizations of multiple structures with automatic clustering

**Data Flow**:
```
Multiple Structure Files → Clustering → Alignment → Individual Projections → Combined Overlay
```

**Key Steps**:
1. Provide multiple structure files via glob patterns: `flatprot overlay "structures/*.cif"`
2. Automatic clustering using Foldseek (optional, can disable with `--no-clustering`)
3. Align structures using family-identity or inertia modes
4. Generate individual projections with opacity scaling based on cluster size
5. Combine into single overlay image (PNG/PDF/SVG)

**Examples**:
```bash
# Basic overlay
flatprot overlay "*.cif" -o overlay.png

# Family-aligned overlay
flatprot overlay "structures/*.cif" --family 3000114 -o family_overlay.pdf

# High-resolution inertia-based overlay
flatprot overlay file1.cif file2.cif file3.cif --alignment-mode inertia --dpi 600 -o overlay.png
```

**Pattern**: Built-in clustering and overlay visualization with configurable alignment strategies

### 6. External Database Integration (`uniprot_projection.py`)
**Use Case**: Fetch structures from external databases and visualize

**Data Flow**:
```
UniProt ID → AlphaFold Database → Structure Download → DSSP Analysis → Projection
```

**Key Steps**:
1. Download structure from AlphaFold DB using UniProt ID
2. Run DSSP for secondary structure assignment: `mkdssp -i structure.cif -o structure.dssp`
3. Project with DSSP: `flatprot project structure.cif --dssp structure.dssp`

**Pattern**: External data integration and processing pipeline

## Core Data Flow Architecture

### Primary Data Transformations
1. **3D Structure Input** → Parse via GEMMI/GemmiStructureParser → `flatprot.core.Structure`
2. **Alignment Phase** → Foldseek search → Transformation matrices → Database lookup
3. **Transformation Phase** → Apply matrices → Transform coordinates
4. **Projection Phase** → Orthographic projection → 2D coordinates
5. **Scene Assembly** → Structure + Annotations → Scene objects
6. **Rendering Phase** → SVG generation → Final output

### Data Types Flow
- **Input**: PDB/CIF files, DSSP files, TOML configuration files
- **Intermediate**: numpy arrays, transformation matrices, scene objects
- **Output**: SVG files, JSON metadata, alignment info

### External Tool Integration
- **Foldseek**: Structure alignment and clustering
- **DSSP (mkdssp)**: Secondary structure assignment
- **GEMMI**: Structure file parsing and domain extraction
- **External APIs**: AlphaFold Database downloads

### Processing Patterns
1. **Single Structure**: Simple linear pipeline
2. **Batch Processing**: Parallel processing of multiple structures
3. **Comparative Analysis**: Alignment-based processing for consistent orientations
4. **Domain-Aware**: Multi-level processing with domain-specific operations
5. **Clustering**: Similarity-based grouping and representative selection
6. **Multi-Structure Overlay**: Automated clustering, alignment, and opacity-scaled combination
