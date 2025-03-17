# DSSP File Format

DSSP (Define Secondary Structure of Proteins) files contain secondary structure assignments for protein structures. These files are required when using PDB format files with FlatProt.

## Overview

DSSP files provide:

-   Secondary structure assignments
-   Solvent accessibility calculations
-   Hydrogen bond information
-   Geometric features

## File Format

DSSP files follow a specific format with both header and data sections.

### Header Section

The header contains information about:

-   The source PDB file
-   DSSP version
-   Processing date
-   Total number of residues
-   Author information

!!!warning
DSSP files are not well specified and can contain errors. We recomend using mmCIF files containing secondary structure information.
FlatProt assumes DSSP files are generated using the [dssp >= 4.4](https://github.com/PDB-REDO/dssp) command line tool.

!!!note
If you supply a dssp file, it will take precedence over the secondary structure information in the mmCIF file.

Example header:

```text
==== Secondary Structure Definition by the program DSSP, CMBI version by M.L. Hekkelman/2010-10-21 ==== DATE=2024-03-15        .
REFERENCE W. KABSCH AND C.SANDER, BIOPOLYMERS 22 (1983) 2577-2637                                                              .
HEADER    PROTEIN                                 01-JAN-20   1ABC                                                              .
COMPND    MOL_ID: 1; MOLECULE: PROTEIN; CHAIN: A;                                                                              .
SOURCE    MOL_ID: 1; ORGANISM_SCIENTIFIC: EXAMPLE ORGANISM; ORGANISM_TAXID: 9999;                                              .
AUTHOR    JOHN DOE                                                                                                             .
```

### Data Section

The data section contains per-residue information in a fixed-width format:

```text
  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
    1    1 A M              0   0  223      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 -34.6   10.3   12.8   13.5
    2    2 A S     >  -     0   0   85      1,-0.1     3,-1.3     0, 0.0     4,-0.3  -0.805 360.0-154.1 -90.5 143.8    9.7   16.5   13.4
```

Fields in order:

1. Residue number
2. Chain identifier
3. Amino acid (one letter code)
4. Secondary structure assignment
5. Beta bridge partner residues
6. Solvent accessibility
7. Hydrogen bond information
8. Geometric parameters (phi, psi angles)
9. CA atom coordinates

## Secondary Structure Codes

DSSP uses the following codes for secondary structure:

| Code | Structure Type | Description                                      |
| ---- | -------------- | ------------------------------------------------ |
| H    | α-helix        | Regular α-helix                                  |
| G    | 3₁₀-helix      | Tighter helix with 3 residues per turn           |
| I    | π-helix        | Wider helix with 5 residues per turn             |
| E    | β-strand       | Extended strand in parallel/antiparallel β-sheet |
| B    | β-bridge       | Isolated β-bridge residue                        |
| T    | Turn           | Hydrogen bonded turn                             |
| S    | Bend           | High curvature region                            |
| ' '  | Coil           | Unstructured region (space character)            |

## Usage with FlatProt

When using PDB format files, a DSSP file must be provided using the `--dssp` option:

```bash
flatprot structure.pdb output.svg --dssp structure.dssp
```

## Validation

FlatProt validates DSSP files for:

1. **Format Compliance**

    - Checks header format
    - Validates data section format
    - Verifies field alignments

2. **Content Validation**

    - Matches residue numbers with structure
    - Validates chain identifiers
    - Checks secondary structure codes

3. **Consistency**
    - Verifies consistency with structure file
    - Checks residue matching
    - Validates chain correspondence

## Error Handling

Common DSSP-related errors:

1. **Missing DSSP File**

    - Error when using PDB without DSSP
    - Solution: Generate DSSP file or use mmCIF

2. **Format Errors**

    - Invalid DSSP format
    - Misaligned columns
    - Invalid secondary structure codes

3. **Consistency Errors**
    - Mismatched residue numbers
    - Inconsistent chain identifiers
    - Structure/DSSP mismatch

## Generating DSSP Files

DSSP files can be generated using the `mkdssp` program:

```bash
mkdssp -i structure.pdb -o structure.dssp
```

For more information about DSSP:

-   [DSSP redo GitHub Manual Page](https://github.com/PDB-REDO/dssp/blob/trunk/doc/mkdssp.md)
-   [DSSP redo Website](https://pdb-redo.eu/dssp/download)
