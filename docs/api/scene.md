# Scene API

This section documents the components related to the FlatProt Scene, which acts as a container for projected visual elements before rendering.

## Scene Concept

The "Scene" in FlatProt acts as a container that holds all the elements to be visualized _after_ they have been projected into 2D space (plus depth information). It's the bridge between the processed structural data and the final rendering step (e.g., SVG generation).

Key concepts:

1.  **Container:** The `Scene` object holds a collection of `SceneElement` objects (like helices, sheets, coils, annotations).
2.  **Coordinate Space:** Elements within the scene typically exist in a 2D coordinate system (X, Y) representing the canvas, but they retain a Z-coordinate representing their depth relative to the viewer.
3.  **Resolution:** It often works with a `CoordinateResolver` to map abstract residue identifiers (like `ChainID:ResidueIndex`) to the actual 2D+Depth coordinates within the scene's context.
4.  **Z-Ordering:** The depth information (Z-coordinate) associated with elements allows renderers to draw them in the correct order, ensuring closer elements obscure farther ones. Annotations are typically given a very high depth value to ensure they are drawn on top.
5.  **Rendering:** The `Scene` object, along with its elements and their associated styles, provides all the necessary information for a `Renderer` (like `SVGRenderer`) to draw the final visualization.

---

## Main Scene Class

The central container for all scene elements.

::: flatprot.scene.Scene
options:
show_root_heading: true
members_order: source

## Base Classes

Abstract base classes for elements and styles within the scene.

::: flatprot.scene.base_element.BaseSceneElement
options:
show_root_heading: true
members_order: source

::: flatprot.scene.base_element.BaseSceneStyle
options:
show_root_heading: true
members_order: source

## Structure Elements

Classes representing secondary structure elements within the scene.

::: flatprot.scene.structure.base_structure.BaseStructureSceneElement
options:
show_root_heading: true

::: flatprot.scene.structure.base_structure.BaseStructureStyle
options:
show_root_heading: true

::: flatprot.scene.structure.helix.HelixSceneElement
options:
show_root_heading: true

::: flatprot.scene.structure.helix.HelixStyle
options:
show_root_heading: true

::: flatprot.scene.structure.sheet.SheetSceneElement
options:
show_root_heading: true

::: flatprot.scene.structure.sheet.SheetStyle
options:
show_root_heading: true

::: flatprot.scene.structure.coil.CoilSceneElement
options:
show_root_heading: true

::: flatprot.scene.structure.coil.CoilStyle
options:
show_root_heading: true

## Annotation Elements

Classes representing annotation elements within the scene.

::: flatprot.scene.annotation.base_annotation.BaseAnnotationElement
options:
show_root_heading: true

::: flatprot.scene.annotation.base_annotation.BaseAnnotationStyle
options:
show_root_heading: true

::: flatprot.scene.annotation.point.PointAnnotation
options:
show_root_heading: true

::: flatprot.scene.annotation.point.PointAnnotationStyle
options:
show_root_heading: true

::: flatprot.scene.annotation.line.LineAnnotation
options:
show_root_heading: true

::: flatprot.scene.annotation.line.LineAnnotationStyle
options:
show_root_heading: true

::: flatprot.scene.annotation.area.AreaAnnotation
options:
show_root_heading: true

::: flatprot.scene.annotation.area.AreaAnnotationStyle
options:
show_root_heading: true

## Coordinate Resolver

Handles the mapping between residue identifiers and scene coordinates.

::: flatprot.scene.resolver.CoordinateResolver
options:
show_root_heading: true
members_order: source

## Scene Utilities

Helper functions for creating and modifying scenes.

::: flatprot.utils.scene_utils.create_scene_from_structure
options:
show_root_heading: true

::: flatprot.utils.scene_utils.add_annotations_to_scene
options:
show_root_heading: true

::: flatprot.utils.domain_utils.create_domain_aware_scene
options:
show_root_heading: true

## Scene Errors

Exceptions specific to scene creation or processing.

::: flatprot.scene.errors
options:
show_root_heading: true
