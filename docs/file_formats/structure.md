# Structure File Format

FlatProt accepts protein structure files in standard formats used in structural biology. This document explains the supported formats and requirements for structure files.

!!!note ""
We recomend supplying a mmCIF file containing secondary structure information.
You can produce a valid mmCIF file containing secondary structure information using [dssp >= 4.4](https://github.com/PDB-REDO/dssp) command line tool.

## Supported Formats

FlatProt supports the following structure file formats:

1. **mmCIF Format** (recommended)

    - File extensions: `.cif`, `.mmcif`
    - Can contain both structure and secondary structure information
    - Preferred format for new structures

2. **PDB Format**
    - File extension: `.pdb`
    - Requires additional DSSP file for secondary structure information
    - Legacy format, still widely used

## File Requirements

### mmCIF Files

-   Must be valid mmCIF format
-   Must contain atomic coordinates
-   Should include secondary structure information
-   Must include chain identifiers
-   Must include residue numbers

!!!note ""
You can produce a valid mmCIF file containing secondary structure information using [dssp >= 4.4](https://github.com/PDB-REDO/dssp) command line tool.

Example header of a valid mmCIF file:

```cif
data_
#
_entry.id   "1ABC"
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
```

### PDB Files

-   Must be valid PDB format
-   Must contain atomic coordinates
-   Must include chain identifiers
-   Must include residue numbers
-   Must have a valid header section
-   Requires a companion DSSP file for secondary structure information

Example header of a valid PDB file:

```pdb
HEADER    PROTEIN                                 01-JAN-20   1ABC
TITLE     EXAMPLE PROTEIN STRUCTURE
COMPND    MOL_ID: 1;
COMPND    MOLECULE: PROTEIN;
COMPND    CHAIN: A;
SOURCE    MOL_ID: 1;
SOURCE    ORGANISM_SCIENTIFIC: EXAMPLE ORGANISM;
SOURCE    ORGANISM_TAXID: 9999;
KEYWDS    PROTEIN, STRUCTURE
EXPDTA    X-RAY DIFFRACTION
AUTHOR    JOHN DOE
REVDAT   1   01-JAN-20 1ABC    0
JRNL        AUTH   J.DOE
REMARK   2
REMARK   2 RESOLUTION.    2.00 ANGSTROMS
```

## Validation

FlatProt performs several validation checks on structure files:

1. **File Existence**

    - Verifies that the specified file exists
    - Checks file permissions

2. **Format Validation**

    - Validates file format based on extension
    - Checks for required sections and headers
    - Verifies coordinate data format

3. **Content Validation**
    - Checks for presence of atomic coordinates
    - Verifies chain identifiers
    - Validates residue numbering
    - Checks for secondary structure information (mmCIF only)

## Error Handling

Common errors and their solutions:

1. **File Not Found**

    - Error: `StructureFileNotFoundError`
    - Solution: Verify file path and permissions

2. **Invalid Format**

    - Error: `InvalidStructureError`
    - Solution: Check file format and content

3. **Missing Secondary Structure**
    - Error: `SecondaryStructureError`
    - Solution: Provide DSSP file for PDB format or use mmCIF

## Best Practices

1. **Use mmCIF Format**

    - More robust and complete than PDB format
    - Includes secondary structure information
    - Better handling of large structures

2. **Include Headers**

    - Provide complete header information
    - Include experimental details
    - Document structure source

3. **Chain Identifiers**

    - Use unique chain identifiers
    - Document chain contents
    - Maintain consistent numbering

4. **Quality Control**
    - Validate coordinates
    - Check for missing atoms
    - Verify secondary structure assignments
