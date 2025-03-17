# FlatProt

<img width="910" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/34d415d3-55be-4784-a96d-035e40fc5afe">

FlatProt is a tool for 2D protein visualization aimed at improving the comparability of protein structures through standardized 2D visualizations. FlatProt focuses on creating highly comparable representations for same-family proteins. In this case FlatProt was used to generate comparable representations for 3 3FTx structures.

## Contents

-   [Getting Started](#getting-started)
-   [First Experiences with Google Colab](#first-experiences-with-google-colab)
-   [Documentation](#documentation)
-   [Data](#data)
-   [Example Visualizations](#example-visualizations)
-   [Domain Annotation File Format](#domain-annotation-file-format)
-   [Runtime for Protein Sizes](#runtime-for-protein-sizes)
-   [Reference](#reference)

## Getting Started

FlatProt needs python version: python= ">=3.11,<=3.13"

### Download FlatProt via uv

We use and recommend [uv](https://github.com/astral-sh/uv) as a fast, reliable Python package installer and resolver.

```shell
# Install FlatProt using uv
uv tool add FlatProt
```

```shell
# Install FlatProt using uv
uvx flatprot
```

### Install Foldseek

Instructions for downloading the Foldseek software can be found in Foldseek's GitHub

-   [Foldseek GitHub](https://github.com/steineggerlab/foldseek)

--> FlatProt needs the path to the foldseek executable passed as argument to the main function to be used by the program.

### Install dssp

Instructions for downloading dssp can be found here:

-   [dssp instructions](https://pdb-redo.eu/dssp/download)

!The program runs on the mkdssp version 4.4.0! (some sources don't provide that (brew does!))

_An example download workflow for usage can also be found in the google colab_

## First experiences with Google Colab

For users to get to know FlatProt we've prepared a Google Colab notebook with a quick tutorial through the major functions/possibilities of FlatProt. Therefore the example protein 1kt1 is used and visualized in different ways. This allows for basic understanding of the functionalities:

-   [Google Colab tutorial](https://colab.research.google.com/drive/17u0twE81kYYspNFsdXUHrCyP33hj0dO6?usp=sharing)

Instructions in the Colab help with the first hands-on.

For direct usage of FlatProt including all parameters the following Collab provides functionalities for using FlatProt without locally downloading anything.
Users can upload their own proteins to the Collab and visualize them as wanted.
The runtime of the Colab is way longer than local usage. Therefore we don't advise using it for big amounts of data:

-   [Google Colab usage](https://colab.research.google.com/drive/1pJHMagKgpTJ1cfBHkBSh2hMlkzQk263d?usp=sharing)

## Documentation

FlatProt's documentation can found here:

-   [FlatProt documentation!](https://t03i.github.io/FlatProt/)

The documentation includes:

-   Detailed CLI usage instructions
-   File format specifications
-   API reference

**important notes:**

-   input PDB files need a header to work (important for predicted structures)
-   FlatProt's methods can be used via command line commands (cli)

## Data

This project uses datasets that can be found on Zenodo. Additional example files can be found here aswell. You can access and download the data using the following link or DOI:

-   [Zenodo Data](https://doi.org/10.5281/zenodo.10674045)

## Example visualizations

### 3FTx: None|Anca_10|Anolis_carolinensis

<img width="409" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/a315fd49-74cc-456e-b0d2-c6b63333b22b">

### 3FTx - Family Overlay

### 1kt1 - domain visualization (family vis)

<img width="1000" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/1f8ac748-09d1-464c-9e42-c6fd93bfeddc">

### 1kt1 - full protein visualization

<img width="450" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/854a89a7-e91c-4ec5-b81d-d33b4f1b96ec">

## Feature Highlights (cystein bonds)

For highlighting features in a structure two types of highlights are possible. Both residue pairs and single residues can be annotated by residue index, name and wanted highlighting color in the following json format:

<img width="464" alt="image" src="https://github.com/user-attachments/assets/84638ff1-fb64-4dad-9b0e-31b18529578d">

Cystein bonds highlight annotations can be created by using FlatProt's "calculate_cystein_bonds" functionality.
An example json can be downloaded here:

## Runtime for protein sizes

<img width="750" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/ff033992-3339-43ab-a7d4-dd71a26dddc4">
<br>
The runtimes are measured on a local lightweight device.<br>

## Reference

In the following document one can find more information on the tool's methodology, result analysis and references of the shown proteins and used software.

[BachelorThesis_ConstantinCarl_Enhancing-Protein-Comparability-with-Standardized-2D-Visualization.pdf](https://github.com/ConstantinCarl/prot2d/files/14605102/BachelorThesis_ConstantinCarl_Enhancing-Protein-Comparability-with-Standardized-2D-Visualization.pdf)
