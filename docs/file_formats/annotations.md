# Annotation File Format

FlatProt uses TOML files for defining annotations to highlight specific features in protein structures. This document explains the format and options available for creating different types of annotations.

## Overview

Annotation files allow you to highlight specific features in your protein structure, including:

-   Individual residues (point annotations)
-   Connections between residues (line/pair annotations)
-   Regions of the structure (area annotations)

## File Format

Annotation files use the [TOML](https://toml.io/) format, with an `annotations` array containing one or more annotation definitions.

### Example Annotation File

```toml
# Point annotation - marks a single residue
[[annotations]]
type = "point"
label = "Active Site"
index = 45
chain = "A"

# Line annotation - connects two residues
[[annotations]]
type = "line"
label = "Disulfide Bond"
indices = [23, 76]
chain = "A"

# Area annotation - highlights a region
[[annotations]]
type = "area"
label = "Binding Domain"
range = { start = 100, end = 150 }
chain = "A"
```

## Annotation Types

### Point Annotations

Point annotations mark individual residues with a symbol.

| Property | Type    | Description                          | Required |
| -------- | ------- | ------------------------------------ | -------- |
| `type`   | String  | Must be `"point"`                    | Yes      |
| `label`  | String  | Descriptive label for the annotation | Yes      |
| `index`  | Integer | Residue index to annotate            | Yes      |
| `chain`  | String  | Chain identifier                     | Yes      |

Example:

```toml
[[annotations]]
type = "point"
label = "Catalytic Residue"
index = 45
chain = "A"
```

### Line/Pair Annotations

Line annotations connect two residues with a line.

| Property  | Type                | Description                          | Required |
| --------- | ------------------- | ------------------------------------ | -------- |
| `type`    | String              | Must be `"line"` or `"pair"`         | Yes      |
| `label`   | String              | Descriptive label for the annotation | Yes      |
| `indices` | Array of 2 Integers | Residue indices to connect           | Yes      |
| `chain`   | String              | Chain identifier                     | Yes      |

Example:

```toml
[[annotations]]
type = "line"
label = "Salt Bridge"
indices = [34, 112]
chain = "A"
```

### Area Annotations

Area annotations highlight a region of the structure.

| Property      | Type    | Description                              | Required |
| ------------- | ------- | ---------------------------------------- | -------- |
| `type`        | String  | Must be `"area"`                         | Yes      |
| `label`       | String  | Descriptive label for the annotation     | Yes      |
| `range`       | Object  | Object with `start` and `end` properties | Yes      |
| `range.start` | Integer | Starting residue index                   | Yes      |
| `range.end`   | Integer | Ending residue index                     | Yes      |
| `chain`       | String  | Chain identifier                         | Yes      |

Example:

```toml
[[annotations]]
type = "area"
label = "Binding Domain"
range = { start = 100, end = 150 }
chain = "A"
```

## Validation

FlatProt validates annotation files to ensure they have the correct format and reference valid residues in the structure. The validation includes:

1. Checking that the TOML syntax is valid
2. Verifying that all required fields are present
3. Ensuring that referenced chains exist in the structure
4. Confirming that referenced residue indices exist in the specified chains
5. Validating that area ranges are valid (start ≤ end)

If an annotation file is invalid, an error message will be displayed explaining the issue.

## Multiple Annotations

You can define multiple annotations of different types in a single file by adding multiple entries to the `annotations` array.

## Styling Annotations

The appearance of annotations is controlled by the style settings. You can customize the appearance by providing a style file with the appropriate annotation style sections.

See the [Style File Format](style.md) documentation for details on how to customize annotation styles.
