<!--
 Copyright 2024 Rostlab.
 SPDX-License-Identifier: Apache-2.0
-->

# About

FlatProt is a Python package for protein structure and sequence analysis. It provides standardized 2D visualizations for enhanced protein comparability, combining efficient processing of proteins with user-friendly visualizations.

## Standard Workflow on the command line

FlatProt generates 2D projections of protein structures in three simple steps:

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

The resulting SVG file contains a clean, publication-ready 2D representation of your protein structure that can be viewed in any web browser or vector graphics editor.

## Key Features

-   Standardized 2D visualization approach for protein analysis
-   Integration with Foldseek alignments for protein family mapping
-   Efficient processing of proteins across varying sizes
-   Fast response times even for large, complex structures
-   Support for proteins up to 3,000 residues
-   Clear and consistent visualizations for structure analysis

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
-   **Domain Analysis**: Support for domain-split visualization with optional domain annotations
-   **LDDT Score Visualization**: Color-coded representation of local distance difference test scores
-   **Residue Annotations**: Optional display of amino acid annotations and chain positions
-   **Custom Highlighting**: Support for manual residue and bond highlighting through JSON configuration

## Citation

> FlatProt: 2D visualization eases protein structure comparison
> Tobias Olenyi, Constantin Carl, Tobias Senoner, Ivan Koludarov, Burkhard Rost
> bioRxiv 2025.04.22.650077; doi: [https://doi.org/10.1101/2025.04.22.650077](https://doi.org/10.1101/2025.04.22.650077)
