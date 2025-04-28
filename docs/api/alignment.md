# Alignment API

This section documents the components responsible for structural alignment in FlatProt, primarily focused on finding the best match in a reference database and retrieving the associated transformation matrix.

## Alignment Concept

The alignment process in FlatProt serves to orient an input protein structure according to a standardized reference frame, typically based on its structural superfamily.

1.  **Structural Search:** It uses an external tool, Foldseek, to search a pre-compiled database of reference structures (e.g., CATH domains) for the best structural match to the input protein.
2.  **Result Filtering:** The Foldseek results are filtered based on metrics like alignment probability (`--min-probability`) or by specifying a direct target ID (`--target-db-id`).
3.  **Matrix Retrieval:** Once a suitable match is identified (represented by a `target_id` from Foldseek), FlatProt queries its internal HDF5 database (`AlignmentDatabase`) using this `target_id`. This database stores pre-calculated 4x4 transformation matrices that map the reference structure (the target) to a standardized orientation for its superfamily.
4.  **Output:** The primary output is the retrieved transformation matrix (`TransformationMatrix`), which can then be used by the `project` command to render the input structure in the standardized orientation. Alignment metadata (scores, matched IDs) can also be saved.

---

## Top-Level Alignment Functions

These functions provide the main entry points for performing alignment using a database.

::: flatprot.alignment
options:
members: - align_structure_database - get_aligned_rotation_database
show_root_heading: true
show_root_toc_entry: false

## Foldseek Interaction

Classes and functions related to running Foldseek and parsing its results.

::: flatprot.alignment.foldseek
options:
show_root_heading: true
show_root_toc_entry: false

## Alignment Database

Class for interacting with the HDF5 alignment database containing pre-calculated matrices and the associated data entry structure.

::: flatprot.alignment.db
options:
show_root_heading: true
show_root_toc_entry: false

## Alignment Utilities

Utility functions used within the alignment module.

::: flatprot.alignment.utils
options:
show_root_heading: true
show_root_toc_entry: false

## Alignment Errors

Exceptions specific to the alignment process.

::: flatprot.alignment.errors
options:
show_root_heading: true
show_root_toc_entry: false
