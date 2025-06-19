<!--
 Copyright 2024 Rostlab.
 SPDX-License-Identifier: Apache-2.0
-->

# About

FlatProt is a Python package for protein structure and sequence analysis. It provides standardized 2D visualizations for enhanced protein comparability, combining efficient processing of proteins with user-friendly visualizations.

## Standard Workflow on the command line

FlatProt provides four main commands for protein structure analysis.

!!!note
The following examples assume you've installed FlatProt with `uv tool add FlatProt`. If you're using the no-install option, replace `flatprot` with `uvx flatprot` in all commands.

### Single Structure Projection

Generate 2D projections of individual protein structures in three simple steps:

1. Obtain the protein structure file (CIF or PDB format)
    - From databases like PDB, AlphaFold, or your own modeling tools
    - CIF format is recommended for better handling of complex structures
2. Calculate secondary structure using DSSP
    - Run: `mkdssp -i structure.cif -o structure.cif`
    - This identifies α-helices, β-sheets, and other structural elements
3. Generate the 2D projection with FlatProt
    - Basic usage: `flatprot project structure.cif -o projection.svg`
    - With pdb file: `flatprot project structure.pdb --dssp structure.dssp -o projection.svg`
    - Customize with options like `--canvas-width 500 --canvas-height 400` for sizing

### Multi-Structure Comparison

Create overlay visualizations to compare multiple related structures:

1. Collect related structure files (same family or similar proteins)
2. Generate family-aligned overlay: `flatprot overlay "structures/*.cif" -o comparison.png`
3. For consistent size comparison: `flatprot overlay "structures/*.cif" --disable-scaling -o overlay.png`

### Structure Alignment

Align structures to known protein families:

1. Find best matching family: `flatprot align structure.cif -i alignment_info.json`
2. Use alignment in projections: `flatprot project structure.cif --matrix alignment_matrix.npy -o aligned.svg`

### Region-Based Analysis

Extract and visualize specific structural regions with comparative alignment:

1. Define regions of interest: `flatprot split protein.cif --regions "A:1-100,A:150-250" -o regions.svg`
2. With database alignment: `flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o aligned_regions.svg`
3. Progressive gap positioning: `flatprot split protein.cif --regions "A:1-100,A:150-250" --gap-x 150 --gap-y 100 -o positioned.svg`

## Common Command Chaining Workflows

### Family Comparison with Optimal Conservation

For optimal conservation across multiple structures, align all to the same family reference:

```bash
# 1. Find optimal alignment for family
flatprot align reference_protein.cif -i family_info.json

# 2. Extract family ID from results
family_id=$(jq -r '.best_hit.target_id' family_info.json)

# 3. Create conserved overlay using fixed family ID
flatprot overlay "family_proteins/*.cif" --family "$family_id" -o family_overlay.png
```

### Aligned Structure Visualization

Create family-aligned 2D projections of individual structures:

```bash
# 1. Find optimal family alignment
flatprot align protein.cif -m alignment_matrix.npy

# 2. Create aligned 2D projection
flatprot project protein.cif --matrix alignment_matrix.npy -o aligned_protein.svg
```

### Comprehensive Domain Analysis

Combine full structure and domain-specific views:

```bash
# 1. Create full structure view
flatprot project protein.cif -o full_structure.svg

# 2. Extract and align specific domains
flatprot split protein.cif --regions "A:1-100,A:150-250" --show-database-alignment -o domains.svg

```


The resulting files contain clean, publication-ready 2D representations that can be viewed in any web browser or vector graphics editor.

## Key Features

-   **Single Structure Visualization**: Standardized 2D projections of individual protein structures
-   **Multi-Structure Overlays**: Compare multiple structures with automatic clustering and alignment
-   **Region-Based Analysis**: Extract and visualize specific domains, motifs, or binding sites with comparative alignment
-   **Family-Based Alignment**: Integration with Foldseek alignments for protein family mapping with rotation-only transformations
-   **Database Annotations**: Automatic SCOP family identification and alignment probability display
-   **Batch Processing**: Efficient processing of proteins across varying sizes
-   **High Performance**: Fast response times even for large, complex structures
-   **Scalable**: Support for proteins up to 3,000 residues and datasets with 50+ structures
-   **Publication Ready**: Clear and consistent visualizations for structure analysis

## Visualization Options

FlatProt offers multiple visualization types to suit different analysis needs:

-   **Normal**: Standard visualization with detailed secondary structure representation
-   **Simple-coil**: Simplified coil representation while maintaining detailed secondary structure elements
-   **Only-path**: Minimalist path representation of the protein structure
-   **Special**: Enhanced visualization with customizable features for specific analysis needs

### Secondary Structure Representation

-   **Helices**: Visualized in red (#F05039), with options for simple or detailed representation
-   **Sheets**: Shown in blue (#1F449C) with arrow representations
-   **Coils**: Displayed in black with adjustable simplification levels

## Advanced Features

-   **Multi-chain Support**: Automatic handling and visualization of multiple protein chains
-   **Region-Specific Extraction**: Precise extraction of structural domains, motifs, and binding sites
-   **Database-Driven Alignment**: Integration with FoldSeek for family-specific structural alignment
-   **Rotation-Only Transformations**: Align region orientations while preserving spatial relationships
-   **SCOP Family Annotations**: Automatic identification and display of structural classification annotations
-   **Alignment Quality Metrics**: Display of alignment probabilities and confidence scores
-   **Progressive Gap Positioning**: Flexible domain spacing with configurable horizontal and vertical gaps for comparative visualization
-   **LDDT Score Visualization**: Color-coded representation of local distance difference test scores
-   **Residue Annotations**: Optional display of amino acid annotations and chain positions
-   **Custom Highlighting**: Support for manual residue and bond highlighting through JSON configuration

## Citation & Data

If you use FlatProt in your research, please cite our preprint:

> **FlatProt: 2D visualization eases protein structure comparison**
> Tobias Olenyi, Constantin Carl, Tobias Senoner, Ivan Koludarov, Burkhard Rost
> *bioRxiv* 2025.04.22.650077; doi: [https://doi.org/10.1101/2025.04.22.650077](https://doi.org/10.1101/2025.04.22.650077)

**Datasets and supplementary data** are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15697296.svg)](https://doi.org/10.5281/zenodo.15697296)
