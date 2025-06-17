# Annotation File Format

FlatProt uses [TOML](https://toml.io/) files for defining annotations to highlight specific features in protein structures. This document explains the format and options available for creating different types of annotations.

## Overview

The annotation file must be a valid TOML document. The primary structure consists of a top-level key named `annotations` which holds an array of tables. Each table in this array represents a single annotation.

Annotations allow you to highlight specific features in your protein structure, including:

-   Individual residues (point annotations)
-   Connections between residues (line annotations)
-   Regions of the structure (area annotations)

You can define multiple annotations in a single file by adding more tables to the `annotations` array.

## File Format

```toml
# This entire file is in TOML format.
# The top-level key must be 'annotations', containing an array of tables.

[[annotations]]

type = "point",          # Required: type of annotation
label = "Active Site His", # Optional: displayed label
index = "A:41",          # Required: target residue (ChainID:ResidueIndex)
id = "point-his41",      # Optional: unique identifier

# Optional inline 'style' table for overrides:
[annotations.style]
    marker_radius = 8.0,
    color = "#FF0000",
    label_offset = [5.0, -5.0] # TOML array for tuple



  # --- Second Annotation Table --- #
[[annotations]]

type = "line",
label = "Disulfide Bond",
indices = ["A:23", "A:76"], # Required: target start/end residues

[annotations.style]
  line_color = "#FFA500",   # Orange
  stroke_width = 2.0,
  line_style = [4.0, 2.0], # Dashed line pattern (TOML array)
  connector_radius = 3.0

  # --- Third Annotation Table --- #
[[annotations]]

type = "area",
label = "Binding Domain",
range = "A:100-150",       # Required: target residue range (ChainID:Start-End)

[annotations.style]
  fill_color = "#00FF00",   # Green
  fill_opacity = 0.25,
  stroke_color = "#808080", # Grey outline
  padding = 15.0

```

## Annotation Table Fields

Each table within the `annotations` list defines a single annotation and must contain the following fields:

| Field   | Type   | Description                                                                             | Required | Default                   |
| ------- | ------ | --------------------------------------------------------------------------------------- | -------- | ------------------------- |
| `type`  | String | Type of annotation. Must be one of: `"point"`, `"line"`, `"area"`.                      | Yes      | N/A                       |
| `label` | String | Optional descriptive text label displayed with the annotation.                          | No       | `None`                    |
| `id`    | String | Optional unique identifier for the annotation. If omitted, one is generated.            | No       | Generated                 |
| `style` | Table  | Optional inline table defining style overrides for this specific annotation. See below. | No       | Default annotation styles |

**Additionally, exactly one of the following targeting fields must be provided:**

| Field     | Type             | Description                                                                 | Required by Type | Format Example     |
| --------- | ---------------- | --------------------------------------------------------------------------- | ---------------- | ------------------ |
| `index`   | String           | Target residue for `point` annotations. Format: `"ChainID:ResidueIndex"`.   | `point` only     | `"A:41"`           |
| `indices` | Array of Strings | Target start and end residues for `line` annotations. Format as `index`.    | `line` only      | `["A:23", "A:76"]` |
| `range`   | String           | Target residue range for `area` annotations. Format: `"ChainID:Start-End"`. | `area` only      | `"A:100-150"`      |

## Inline Style (`style` Table)

Each annotation can optionally include a `style` table (a TOML table) to override default appearance settings. If a `style` table is present, its fields override the corresponding defaults for that specific annotation type. If the `style` table or individual fields within it are omitted, the default values are used.

### Common Style Attributes (Applicable to all types within the `style` table)

| Attribute           | Type                | Default      | Description                                                    |
| ------------------- | ------------------- | ------------ | -------------------------------------------------------------- |
| `color`             | Color               | `"#FF0000"`  | Default color (e.g., for marker fill, line stroke). Red.       |
| `offset`            | Array[Float, Float] | `[0.0, 0.0]` | 2D offset (x, y) applied to the annotation's anchor point.     |
| `label_color`       | Color               | `"#000000"`  | Color for the annotation label (default black).                |
| `label_font_size`   | Float               | `12.0`       | Font size for the label.                                       |
| `label_font_weight` | String              | `"normal"`   | Font weight for the label (e.g., "normal", "bold").            |
| `label_font_family` | String              | `"Arial"`    | Font family for the label.                                     |
| `label_offset`      | Array[Float, Float] | `[0.0, 0.0]` | 2D offset (x, y) applied specifically to the label's position. |

### Point Annotation Style (`style` table when `type = "point"`)

| Attribute       | Type  | Default | Description                                        |
| --------------- | ----- | ------- | -------------------------------------------------- |
| `marker_radius` | Float | `5.0`   | Radius (size) of the point marker (must be >= 0). |

Note: Point markers are always rendered as circles.

### Line Annotation Style (`style` table when `type = "line"`)

| Attribute          | Type              | Default      | Description                                                           |
| ------------------ | ----------------- | ------------ | --------------------------------------------------------------------- |
| `stroke_width`     | Float             | `1.0`        | Width of the annotation line (must be >= 0).                          |
| `line_style`       | Array[Float, ...] | `[5.0, 5.0]` | Dash pattern (e.g., `[5.0, 5.0]` for dashed). Empty `[]` means solid. |
| `connector_color`  | Color             | `"#000000"`  | Color of the small circles at the start/end points (default black).   |
| `line_color`       | Color             | `"#000000"`  | Color of the line itself (default black).                             |
| `arrowhead_start`  | Boolean           | `false`      | Draw an arrowhead at the start of the line?                           |
| `arrowhead_end`    | Boolean           | `false`      | Draw an arrowhead at the end of the line?                             |
| `connector_radius` | Float             | `2.0`        | Radius of the connector circles (must be >= 0).                       |

### Area Annotation Style (`style` table when `type = "area"`)

| Attribute              | Type              | Default | Description                                                                     |
| ---------------------- | ----------------- | ------- | ------------------------------------------------------------------------------- |
| `fill_color`           | Color             | `None`  | Fill color. If `None`, uses `color` with `fill_opacity`.                        |
| `fill_opacity`         | Float             | `0.3`   | Opacity for the fill (0.0 to 1.0).                                              |
| `stroke_width`         | Float             | `1.0`   | Width of the area outline (must be >= 0).                                       |
| `linestyle`            | Array[Float, ...] | `[]`    | Dash pattern for the outline. Empty `[]` means solid.                           |
| `padding`              | Float             | `20.0`  | Padding pixels added outside the convex hull of the area points (must be >= 0). |
| `interpolation_points` | Integer           | `3`     | (Internal detail related to rendering, might not be useful to expose widely)    |
| `smoothing_window`     | Integer           | `1`     | (Internal detail related to rendering, might not be useful to expose widely)    |

## Color Formats

Colors can be specified in any of the following formats recognized by Pydantic's `Color` type:

-   Hex codes: `"#FF5733"`, `"FF5733"`, `"#F53"`, `"F53"`
-   RGB format: `"rgb(255, 87, 51)"`
-   RGBA format: `"rgba(255, 87, 51, 0.5)"` (Alpha is usually handled by `opacity` or `fill_opacity` style fields)
-   Named colors: `"red"`, `"blue"`, `"green"`, etc. (Standard web color names)

## Validation

FlatProt validates annotation files when parsing:

1.  Checks for valid TOML syntax.
2.  Ensures the top-level `annotations` list exists and contains tables (dictionaries).
3.  Verifies that each annotation table has a valid `type`.
4.  Checks for the presence and correct format of the required targeting field (`index`, `indices`, or `range`).
5.  Validates the format of coordinate and range strings (`ChainID:Index`, `ChainID:Start-End`).
6.  If an inline `style` table is present, validates its fields against the corresponding style model (e.g., checks types, ranges, allowed values).

If an annotation file is invalid, an error message will be displayed explaining the specific issue and location (e.g., `Annotation #3: Invalid value for 'marker_radius'`).
