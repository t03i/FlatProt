# FlatProt

<img width="910" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/34d415d3-55be-4784-a96d-035e40fc5afe">

FlatProt is a tool for 2D protein visualization aimed at improving the comparability of protein structures through standardized 2D visualizations. FlatProt focuses on creating highly comparable representations for same-family proteins. In this case FlatProt was used to generate comparable representatoins for 3 3FTx structures.

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

FlatProt needs python version: python= ">=3.10,<3.13"

### Download FlatProt via pip or poetry

-   pip:

```shell
pip install FlatProt
```

-   Poetry:

```shell
poetry add FlatProt
```

### Install Foldseek

Instructions for downloading the Foldseek software can be found in Foldseek's GitHub

-   [Foldseek GitHub](https://github.com/steineggerlab/foldseek)

--> FlatProt needs the path to the foldseek executable passed as argument to the main function to be used by the program.

### Install dssp

Instructions for downloading dssp can be found here:

-   [dssp instructions](http://biskit.pasteur.fr/install/applications/dssp)

!The program runs on the mkdssp version 4.4.0! (some sources dont provide that (brew does!))

_An example download workflow for usage can also be found in the google colab_

## First experiences with Google Colab

For users to get to know prot2d we've prepared a Google Colab notebook with a quick tutorial through the major functions/possibilities of prot2d. Therefore the exmample protein 1kt1 is used and visualized in different ways. This allows for basic understanding of the functionalities:

-   [Google Colab tutorial](https://colab.research.google.com/drive/17u0twE81kYYspNFsdXUHrCyP33hj0dO6?usp=sharing)

Instructions in the Colab help with the first hands-on.

For direct usage of prot2d including all paramters the following Collab provides functionalitys for using prot2d withput locally downloading anything.
Users can upload theri own proteins to the Collab and visualize them as wanted.
The runtime of the Collab is way longer than local usage. Therefore we dont advise using it for big amounts of data:

-   [Google Colab usage](https://colab.research.google.com/drive/1pJHMagKgpTJ1cfBHkBSh2hMlkzQk263d?usp=sharing)

## Documentation

FlatProt's documentation can found here:

-   [FlatProt documentation!](https://constantincarl.github.io/FlatProt/)

**important notes:**

-   input PDB files need a header to work (important for predicted structures)
-   FlatProt's methods can also be used via command line commands (cli)

## Data

This project uses datasets that can be found on Zenodo. Additional example files can be found here aswell. You can access and download the data using the following link or DOI:

-   [Zenodo Data](https://doi.org/10.5281/zenodo.10674045)

## Example visualizations

### 3FTx: None|Anca_10|Anolis_carolinensis

<img width="409" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/a315fd49-74cc-456e-b0d2-c6b63333b22b">

### 3FTx - Family Overlay


### 1kt1 - domain visualization (family vis)

<img width="1000" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/1f8ac748-09d1-464c-9e42-c6fd93bfeddc">

### 1kt1 - full protine visualization

<img width="450" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/854a89a7-e91c-4ec5-b81d-d33b4f1b96ec">

## Domain Annotation File Format:

The domain annotation is needed in the following format to be procceced by prot2d (methods for converting chainsaw annotations are included in the package documentation)

![image](https://github.com/ConstantinCarl/prot2d/assets/156075124/54dd63d9-933f-4358-8eae-8aa838008de5)

## Runtime for protein sizes

<img width="750" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/ff033992-3339-43ab-a7d4-dd71a26dddc4">
<br>
The runtimes are measured on a local lightweight device.<br>

## Reference

In the following document one can find more information on the tools methodogy, result analysis and references of the shown proteins and used software.

[BachelorThesis_ConstantinCarl_Enhancing-Protein-Comparability-with-Standardized-2D-Visualization.pdf](https://github.com/ConstantinCarl/prot2d/files/14605102/BachelorThesis_ConstantinCarl_Enhancing-Protein-Comparability-with-Standardized-2D-Visualization.pdf)
