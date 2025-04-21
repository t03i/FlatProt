# Transformation & Projection API

This section documents functions and classes related to transforming protein structures (e.g., alignment, inertia-based) and projecting them into 2D coordinates for visualization.

## Transformation Matrix

Defines the core class for handling 3D transformation matrices.

::: flatprot.transformation.TransformationMatrix
options:
show_root_heading: true
members_order: source

## Inertia-Based Transformation

Classes and functions for calculating and applying transformations based on the principal axes of inertia.
These transformations are often used for aligning protein structures in a way that preserves their overall shape and orientation.
Commonly these are implemented in orientation features of molecular visualization software.

::: flatprot.transformation.inertia_transformation.calculate_inertia_transformation_matrix
options:
show_root_heading: true

::: flatprot.transformation.inertia_transformation.InertiaTransformationParameters
options:
show_root_heading: true
members_order: source

::: flatprot.transformation.inertia_transformation.InertiaTransformer
options:
show_root_heading: true
members_order: source

## Matrix-Based Transformation

Classes for applying a pre-defined transformation matrix.

::: flatprot.transformation.matrix_transformation.MatrixTransformParameters
options:
show_root_heading: true
members_order: source

::: flatprot.transformation.matrix_transformation.MatrixTransformer
options:
show_root_heading: true
members_order: source

## Structure Transformation Utilities

Helper functions (often used internally by CLI commands) for applying transformations to `flatprot.core.Structure` objects.

::: flatprot.utils.structure_utils.transform_structure_with_inertia
options:
show_root_heading: true

::: flatprot.utils.structure_utils.transform_structure_with_matrix
options:
show_root_heading: true

## Orthographic Projection

Function for projecting a transformed 3D structure onto a 2D canvas.

::: flatprot.utils.structure_utils.project_structure_orthographically
options:
show_root_heading: true

## Domain Transformation Utilities

Classes and functions specifically for handling domain-based transformations.

::: flatprot.utils.domain_utils.DomainTransformation
options:
show_root_heading: true
members_order: source

::: flatprot.utils.domain_utils.apply_domain_transformations_masked
options:
show_root_heading: true
