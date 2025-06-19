# Rendering API

This section documents the rendering system in FlatProt, responsible for converting the abstract `Scene` object into a concrete visual output format, primarily SVG.

## Rendering Concept

Renderers act as the final stage in the visualization pipeline. They take a populated `Scene` object, which contains various `SceneElement` instances (like `HelixSceneElement`, `PointAnnotation`, etc.) already projected into a 2D canvas space with associated depth (Z) information.

The renderer iterates through the elements in the scene, translates their geometric data and style attributes into the target output format (e.g., SVG tags and attributes), and produces the final file or output string.

## Peculiarities and Design Choices

Several aspects define how the FlatProt rendering system, particularly the `SVGRenderer`, operates:

1.  **Depth Sorting (Z-Ordering):**

    -   Before drawing, scene elements are typically sorted based on their calculated average depth (Z-coordinate). This ensures that elements closer to the viewer (lower Z, assuming standard projection) are drawn later, correctly occluding elements farther away.
    -   Annotation elements (`BaseAnnotationElement`) usually override the depth calculation to return a very high value (e.g., `float('inf')`). This guarantees they are sorted last and therefore drawn _on top_ of all structural elements.

2.  **Scene Element to SVG Mapping:**

    -   The renderer maps different `SceneElement` types to appropriate SVG tags:
        -   `HelixSceneElement`: Typically rendered as an SVG `<path>` or `<polygon>` representing the zigzag ribbon.
        -   `SheetSceneElement`: Rendered as an SVG `<polygon>` forming the arrowhead shape.
        -   `CoilSceneElement`: Rendered as an SVG `<path>` or `<polyline>` representing the smoothed line.
        -   `PointAnnotation`: Rendered as an SVG `<circle>` plus an SVG `<text>` element for the label.
        -   `LineAnnotation`: Rendered as an SVG `<line>` or `<path>`, potentially with `<circle>` elements for connectors and `<polygon>` for arrowheads, plus an SVG `<text>` element for the label.
        -   `AreaAnnotation`: Rendered as an SVG `<path>` or `<polygon>` representing the padded convex hull, plus an SVG `<text>` element for the label.
    -   Elements are often grouped within SVG `<g>` tags for organization, potentially grouped by type or parent element in the scene graph.

3.  **Style Application:**

    -   Style attributes defined in the `BaseSceneStyle` and its derivatives (e.g., `HelixStyle`, `PointAnnotationStyle`) are translated into SVG presentation attributes.
    -   Examples:
        -   `color` or `fill_color` -> `fill` attribute.
        -   `stroke_color` or `line_color` -> `stroke` attribute.
        -   `stroke_width` -> `stroke-width` attribute.
        -   `opacity` or `fill_opacity` -> `opacity` or `fill-opacity` attributes.
        -   `line_style` or `linestyle` (tuple/array) -> `stroke-dasharray` attribute.
        -   Label styles (`label_color`, `label_font_size`, etc.) -> corresponding attributes on the `<text>` element.

4.  **Coordinate System & Canvas:**

    -   The input `Scene` contains elements with coordinates already projected onto the 2D canvas (X, Y) plus depth (Z). The origin (0,0) is typically the top-left corner, consistent with SVG standards.
    -   The `width` and `height` provided to the renderer define the dimensions of the SVG canvas and its `viewBox`. These dimensions are controlled via CLI parameters (default: 1000x1000) and passed through the rendering pipeline.
    -   The `project_structure_orthographically` utility function handles the scaling and centering of the protein coordinates within this canvas space before they reach the scene/renderer.

5.  **Focus on Static Output:**
    -   The current implementation focuses on generating static SVG images. It generally does not utilize advanced SVG features like animations, complex gradients, filters, or embedded scripts.

---

## Renderer Classes

::: flatprot.renderers.svg_renderer.SVGRenderer
options:
show_root_heading: true
members_order: source
