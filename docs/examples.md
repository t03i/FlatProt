# Examples

This page provides brief descriptions and links to run the example notebooks found in the `examples/` directory.

## `uniprot_projection.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/uniprot_projection.ipynb)

This notebook demonstrates how to fetch a protein structure from the AlphaFold Database (AFDB) using a UniProt ID, run DSSP to determine secondary structure, and then generate a 2D projection using FlatProt.

## `annotations.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/annotations.ipynb)

This notebook demonstrates how to apply custom styles and various annotations (point, line, area) to a FlatProt projection.

## `3ftx_alignment.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/3ftx_alignment.ipynb)

This notebook demonstrates aligning three related three-finger toxin structures (Cobra, Krait, Snake) using a Foldseek database and then projecting them into 2D SVG visualizations using FlatProt.

## `chainsaw.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/chainsaw.ipynb)

Generate and compare different 2D SVG visualizations of a protein structure based on its domains: Normal Projection, Domain-Aligned Projection, Domain-Separated Projection.

## `dysulfide.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/dysulfide.ipynb)

This notebook demonstrates how to: Compute cystine (disulfide) bridges from a protein structure file (`.cif`); Create a FlatProt annotation file (`.toml`) highlighting these bridges; Generate a 2D SVG projection of the protein using `flatprot project`, applying the annotations.

## `klk_overlay.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/klk_overlay.ipynb)

This example script demonstrates how to align KLK (Kallikrein) structures and overlay their FlatProt projections.

## `runtime.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/t03i/FlatProt/blob/notebooks/examples/runtime.ipynb)

Runtime measurement script for FlatProt alignment and projection on human proteome. This script measures the runtime of FlatProt's structural alignment and projection functionality on AlphaFold predicted structures from the human proteome.
