# Input/Output API

This section documents the components responsible for handling file input and output in FlatProt, including parsing structure files and configuration files (styles, annotations).

## IO Concept

The IO module acts as the interface between FlatProt's internal data structures and external files. Its primary responsibilities are:

1.  **Structure Parsing:** Reading 3D coordinates, sequence information, and potentially secondary structure assignments from standard formats like PDB and mmCIF. This often involves leveraging libraries like Gemmi (e.g., via `GemmiStructureParser`).
2.  **Configuration Parsing:** Reading and validating configuration files written in TOML format, specifically for custom styles (`StyleParser`) and annotations (`AnnotationParser`). These parsers translate the TOML definitions into structured Pydantic models used by the Scene and Rendering systems.
3.  **Validation:** Performing basic checks on input files (e.g., existence, basic format validation) before attempting full parsing.
4.  **Error Handling:** Defining specific exception types related to file reading, parsing, and validation errors.

---

## Structure Parser

Handles reading and parsing protein structure files (PDB, mmCIF).

::: flatprot.io.GemmiStructureParser
options:
show_root_heading: true
members_order: source

## Style Parser

Parses TOML files defining custom styles for structure elements.

::: flatprot.io.styles.StyleParser
options:
show_root_heading: true
members_order: source

## Annotation Parser

Parses TOML files defining annotations (points, lines, areas) and their inline styles.

::: flatprot.io.annotations.AnnotationParser
options:
show_root_heading: true
members_order: source

## File Validation

Utility functions for validating input files.

::: flatprot.io.validate_structure_file
options:
show_root_heading: true

::: flatprot.io.validate_optional_files
options:
show_root_heading: true

## IO Errors

Exceptions specific to file input, output, parsing, and validation.

::: flatprot.io.errors
options:
show_root_heading: true
