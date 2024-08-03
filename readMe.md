# prot2d
<img width="910" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/34d415d3-55be-4784-a96d-035e40fc5afe">

Prot2d is a tool for 2D protein visualization aimed at improving the comparability of protein structures through standardized 2D visualizations. Prot2d focuses on creating highly comparable representations for same-family proteins.


# Getting Started

prot2d needs python version: python= ">=3.10,<3.13"


## Download prot2d via pip or poetry
- pip: 
```shell
pip install prot2d
```
- Poetry: 
```shell
poetry add prot2d
```


## Install Foldseek 
Instructions for downloading the Foldseek software can be found in Foldseek's GitHub
- [Foldseek GitHub](https://github.com/steineggerlab/foldseek)

--> prot2d needs the path to the foldseek executable passed as argument to the main function to be used by the program.

## Install dssp
Instructions for downloading dssp can be found here:
- [dssp instructions](http://biskit.pasteur.fr/install/applications/dssp)

!The program runs on the mkdssp version 4.4.0! (some sources dont provide that (brew does!))

*An example download workflow for usage can also be found in the google colab*

# First experiences with Google Colab

For users to get to know prot2d we've prepared a Google Colab notebook. This allows you to execute different prot2d's functionality on example structure and see the results without needing to set up the project locally.

- [Google Colab](https://colab.research.google.com/drive/17u0twE81kYYspNFsdXUHrCyP33hj0dO6?usp=sharing)

Instructions in the Colab help with the first hands-on.


# Documentation

Prot2d's documentation can found here:
- [prot2d documentation!](https://constantincarl.github.io/prot2d/)

**important notes:**
- input PDB files need a header to work (important for predicted structures)
- prot2d's methods can also be used via command line commands (cli)

# Data

This project uses datasets that can be found on Zenodo. Additional example files can be found here aswell. You can access and download the data using the following link or DOI:

- [Zenodo Data](https://doi.org/10.5281/zenodo.10674045)


# Example visualizations

## 1kt1 - domain visualization (family vis)
<img width="1351" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/1f8ac748-09d1-464c-9e42-c6fd93bfeddc">

## 1kt1 - full protine visualization
<img width="540" alt="image" src="https://github.com/ConstantinCarl/prot2d/assets/156075124/854a89a7-e91c-4ec5-b81d-d33b4f1b96ec">









# Domain Annotation File Format:

The domain annotation is needed in the following format to be procceced by prot2d (methods for converting chainsaw annotations are included in the package documentation)

![image](https://github.com/ConstantinCarl/prot2d/assets/156075124/54dd63d9-933f-4358-8eae-8aa838008de5)


