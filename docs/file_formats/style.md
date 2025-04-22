# Style File Format

FlatProt uses TOML files for defining custom styles for protein visualizations. This document explains the format and options available for styling different elements of the visualization.

## Overview

Style files allow you to customize the appearance of various elements in the FlatProt visualization, including:

-   Secondary structure elements (helices, sheets, coils)
-   Annotations (points, lines, areas)
-   Canvas properties

## File Format

Style files use the [TOML](https://toml.io/) format, with each section defining styles for a specific element type.

### Example Style File

```toml
# Canvas settings
[canvas]
width = 800
height = 600
background_color = "#FFFFFF"
margin = 20

# Secondary structure styles
[helix]
fill_color = "#FF5733"
stroke_color = "#000000"
stroke_width = 1.5
amplitude = 0.5

[sheet]
fill_color = "#33FF57"
stroke_color = "#000000"
stroke_width = 1.0
min_sheet_length = 3

[coil]
stroke_color = "#888888"
stroke_width = 1.0

# Annotation styles
[point_annotation]
fill_color = "#3357FF"
stroke_color = "#000000"
radius = 5.0

[line_annotation]
stroke_color = "#FF33A8"
stroke_width = 2.0

[area_annotation]
fill_color = "#FFFF33"
fill_opacity = 0.5
stroke_color = "#000000"
stroke_width = 1.0
padding = 2.0
smoothing_window = 3
interpolation_points = 10
```

## Style Sections

### Canvas Settings

The `[canvas]` section defines the overall properties of the visualization.

| Property           | Type    | Description                               |
| ------------------ | ------- | ----------------------------------------- |
| `width`            | Integer | Width of the canvas in pixels             |
| `height`           | Integer | Height of the canvas in pixels            |
| `background_color` | Color   | Background color of the canvas            |
| `margin`           | Integer | Margin around the visualization in pixels |

### Secondary Structure Styles

These sections define the appearance of secondary structure elements rendered in the visualization. Common properties like `color`, `stroke_color`, `stroke_width`, and `opacity` are inherited but can be overridden.

#### Helix Style (`[helix]`)

Defines the appearance of alpha helices, typically rendered as zigzag ribbons.

| Property           | Type    | Default     | Description                                                                           |
| ------------------ | ------- | ----------- | ------------------------------------------------------------------------------------- |
| `color`            | Color   | `"#ff0000"` | Fill color for the helix ribbon (default red).                                        |
| `stroke_color`     | Color   | `"#000000"` | Color for the outline/stroke (default black).                                         |
| `stroke_width`     | Float   | `1.0`       | Reference width; used as a base for other dimensions.                                 |
| `opacity`          | Float   | `1.0`       | Opacity of the helix element (0.0 to 1.0).                                            |
| `ribbon_thickness` | Float   | `8.0`       | Thickness of the zigzag ribbon.                                                       |
| `wavelength`       | Float   | `10.0`      | Length of one full zigzag cycle along the helix axis.                                 |
| `amplitude`        | Float   | `3.0`       | Height of the zigzag peaks/valleys from the center line.                              |
| `min_helix_length` | Integer | `4`         | Minimum number of residues required to draw a zigzag shape instead of a simple line.  |
| `simplified_width` | Float   | `2.0`       | Line width used when the helix is rendered as a simple line (below min_helix_length). |

#### Sheet Style (`[sheet]`)

Defines the appearance of beta sheets, typically rendered as arrows.

| Property           | Type    | Default     | Description                                                                           |
| ------------------ | ------- | ----------- | ------------------------------------------------------------------------------------- |
| `color`            | Color   | `"#0000ff"` | Fill color for the sheet arrow (default blue).                                        |
| `stroke_color`     | Color   | `"#000000"` | Color for the outline/stroke (default black).                                         |
| `stroke_width`     | Float   | `1.0`       | Reference width; primarily defines the base width of the arrow body.                  |
| `opacity`          | Float   | `1.0`       | Opacity of the sheet element (0.0 to 1.0).                                            |
| `arrow_width`      | Float   | `8.0`       | Width of the arrowhead base relative to the start point.                              |
| `min_sheet_length` | Integer | `3`         | Minimum number of residues required to draw an arrow shape instead of a line.         |
| `simplified_width` | Float   | `2.0`       | Line width used when the sheet is rendered as a simple line (below min_sheet_length). |

#### Coil Style (`[coil]`)

Defines the appearance of coil regions, typically rendered as smoothed lines.

| Property           | Type  | Default     | Description                                                                                                      |
| ------------------ | ----- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `color`            | Color | `"#5b5859"` | Color used for the coil line (default light grey). This often isn't visible as coils usually only have a stroke. |
| `stroke_color`     | Color | `"#000000"` | Color for the coil line (default black).                                                                         |
| `stroke_width`     | Float | `1.0`       | Width of the coil line.                                                                                          |
| `opacity`          | Float | `1.0`       | Opacity of the coil line (0.0 to 1.0).                                                                           |
| `smoothing_factor` | Float | `0.1`       | Fraction of points to keep during smoothing (0.0=max smoothing, 1.0=no smoothing).                               |

### Annotation Styles

#### Point Annotation Style

The `[point_annotation]` section defines the appearance of point annotations.

| Property       | Type  | Description                     |
| -------------- | ----- | ------------------------------- |
| `fill_color`   | Color | Fill color for point markers    |
| `stroke_color` | Color | Outline color for point markers |
| `radius`       | Float | Radius of point markers         |

#### Line Annotation Style

The `[line_annotation]` section defines the appearance of line annotations.

| Property       | Type  | Description                |
| -------------- | ----- | -------------------------- |
| `stroke_color` | Color | Color for line annotations |
| `stroke_width` | Float | Width of line annotations  |

#### Area Annotation Style

The `[area_annotation]` section defines the appearance of area annotations.

| Property               | Type    | Description                                |
| ---------------------- | ------- | ------------------------------------------ |
| `fill_color`           | Color   | Fill color for area annotations            |
| `fill_opacity`         | Float   | Opacity for area fills (0.0-1.0)           |
| `stroke_color`         | Color   | Outline color for areas                    |
| `stroke_width`         | Float   | Width of the area outline                  |
| `padding`              | Float   | Padding around the area                    |
| `smoothing_window`     | Integer | Window size for smoothing the area outline |
| `interpolation_points` | Integer | Number of points to use for interpolation  |

## Color Formats

Colors can be specified in any of the following formats:

-   Hex codes: `"#FF5733"` (with or without the `#` prefix)
-   RGB format: `"rgb(255, 87, 51)"`
-   RGBA format: `"rgba(255, 87, 51, 0.5)"`
-   Named colors: `"red"`, `"blue"`, `"green"`, etc.

## Default Styles

If a style file is not provided, or if certain properties are omitted, FlatProt will use default styles. You only need to specify the properties you want to customize.

## Validation

FlatProt validates style files to ensure they have the correct format and property types. If a style file is invalid, an error message will be displayed explaining the issue.
