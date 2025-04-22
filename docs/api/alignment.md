# Alignment API

This section documents the components responsible for structural alignment in FlatProt, primarily focused on finding the best match in a reference database and retrieving the associated transformation matrix.

## Alignment Concept

The alignment process in FlatProt serves to orient an input protein structure according to a standardized reference frame, typically based on its structural superfamily.

1.  **Structural Search:** It uses an external tool, Foldseek, to search a pre-compiled database of reference structures (e.g., CATH domains) for the best structural match to the input protein.
2.  **Result Filtering:** The Foldseek results are filtered based on metrics like alignment probability (`--min-probability`) or by specifying a direct target ID (`--target-db-id`).
3.  **Matrix Retrieval:** Once a suitable match is identified (represented by a `target_id` from Foldseek), FlatProt queries its internal HDF5 database (`AlignmentDatabase`) using this `target_id`. This database stores pre-calculated 4x4 transformation matrices that map the reference structure (the target) to a standardized orientation for its superfamily.
4.  **Output:** The primary output is the retrieved transformation matrix (`TransformationMatrix`), which can then be used by the `project` command to render the input structure in the standardized orientation. Alignment metadata (scores, matched IDs) can also be saved.

---

## Core Alignment Functions

These functions orchestrate the alignment process.

::: flatprot.alignment.align_structure_database
options:
show_root_heading: true

::: flatprot.alignment.get_aligned_rotation_database
options:
show_root_heading: true

## Database Interaction

Class for interacting with the HDF5 alignment database containing pre-calculated matrices.

::: flatprot.alignment.AlignmentDatabase
options:
show_root_heading: true
members_order: source

## Data Structures

Classes representing alignment results and database entries.

::: flatprot.alignment.AlignmentResult
options:
show_root_heading: true

::: flatprot.alignment.DatabaseEntry
options:
show_root_heading: true

## Alignment Errors

Exceptions specific to the alignment process.

::: flatprot.alignment.errors
options:
show_root_heading: true
