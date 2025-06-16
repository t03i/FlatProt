# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Annotated

from cyclopts import Parameter

from flatprot.core import (
    FlatProtError,
    logger,
)
from flatprot.io import (
    validate_structure_file,
    validate_optional_files,
    GemmiStructureParser,
    StyleParser,
)
from flatprot.utils.structure_utils import (
    transform_structure_with_matrix,
    transform_structure_with_inertia,
    project_structure_orthographically,
)
from flatprot.utils.scene_utils import (
    create_scene_from_structure,
    add_annotations_to_scene,
    add_position_annotations_to_scene,
)
from flatprot.renderers import SVGRenderer
from .errors import error_handler
from .utils import set_logging_level, CommonParameters, print_success_summary


@error_handler
def project_structure_svg(
    structure: Path,
    output: Annotated[Optional[Path], Parameter(name=["-o", "--output"])] = None,
    matrix: Optional[Path] = None,
    style: Optional[Path] = None,
    annotations: Optional[Path] = None,
    dssp: Optional[Path] = None,
    canvas_width: int = 1000,
    canvas_height: int = 1000,
    show_positions: str = "minimal",
    *,
    common: CommonParameters | None = None,
) -> int:
    """
    Generate a 2D projection of a protein structure.

    This function processes a protein structure file, applies transformations, styles, and annotations, and generates a 2D SVG visualization.

    Args:
        structure: Path to the structure file (PDB or similar).
            Supported formats include PDB (.pdb) and mmCIF (.cif, .mmcif).
            The file must exist and be in a valid format.
        output: Path to save the SVG output.
            If not provided, the SVG is printed to stdout.
            The directory will be created if it doesn't exist.
        matrix: Path to a custom transformation matrix.
            If not provided, a default inertia transformation is used.
            The matrix should be a NumPy (.npy) file containing (a) A 4x3 matrix where the first 3 rows are the rotation matrix (3x3) and the last row is the translation vector (1x3) (b) Alternatively, a 3x3 rotation matrix (translation will be set to zero) (c) The matrix can also be transposed (3x4) and will be automatically corrected
        style: Path to a custom style file in TOML format.
            If not provided, the default styles are used.
            The style file can define visual properties for helices, sheets,
            points, lines, and areas. See examples/styles.toml for reference.
        annotations: Path to a TOML file with annotation definitions.
            The annotation file can define point, line, and area annotations to highlight
            specific structural features. Examples:
            - Point annotations mark single residues with symbols
            - Line annotations connect residues with lines
            - Area annotations highlight regions of the structure
            See examples/annotations.toml for a reference annotation file.
        dssp: Path to a DSSP file with secondary structure assignments.
            If not provided, secondary structure is assumed to be in the input structure file.
            Required for PDB files, as they don't contain secondary structure information.
        canvas_width: Width of the canvas in pixels.
        canvas_height: Height of the canvas in pixels.
        show_positions: Position annotation level controlling residue numbering and terminus labels.
            Available levels:
            - 'none': No position annotations
            - 'minimal': Only N/C terminus labels (default)
            - 'major': N/C terminus + residue numbers for major secondary structures (≥3 residues)
            - 'full': All position annotations including short structures
    Returns:
        int: 0 for success, 1 for errors.
    Examples:
        Basic usage:
            flatprot structure.pdb output.svg
        With custom styles and annotations:
            flatprot structure.cif output.svg --annotations annotations.toml --style style.toml
        Using a custom transformation matrix:
            flatprot structure.pdb output.svg --matrix custom_matrix.npy
        Providing secondary structure information for PDB files:
            flatprot structure.pdb output.svg --dssp structure.dssp
        Position annotation examples:
            flatprot structure.cif output.svg --show-positions none
            flatprot structure.cif output.svg --show-positions minimal
            flatprot structure.cif output.svg --show-positions major
            flatprot structure.cif output.svg --show-positions full

    """
    set_logging_level(common)

    # Validate show_positions parameter
    valid_position_levels = {"none", "minimal", "major", "full"}
    if show_positions not in valid_position_levels:
        raise FlatProtError(
            f"Invalid position annotation level: {show_positions}. "
            f"Must be one of {valid_position_levels}"
        )

    try:
        # Validate the structure file
        validate_structure_file(structure)

        # Check if secondary structure information can be extracted
        is_cif_file = structure.suffix.lower() in (".cif", ".mmcif")
        if not is_cif_file and dssp is None:
            raise FlatProtError(
                "Secondary structure information cannot be extracted from non-CIF files. "
                "Please provide either:\n"
                "  - A structure file in CIF format (.cif, .mmcif), or\n"
                "  - A DSSP file using the --dssp option\n"
                "Example: flatprot structure.pdb output.svg --dssp structure.dssp"
            )

        # Verify optional files if specified
        validate_optional_files([matrix, style, annotations, dssp])

        # Load structure
        structure_obj = GemmiStructureParser().parse_structure(structure, dssp)
        if (
            hasattr(structure_obj, "coordinates")
            and structure_obj.coordinates is not None
        ):
            logger.debug(
                f"Coordinates shape after load: {structure_obj.coordinates.shape}"
            )
        else:
            logger.debug("Coordinates attribute not found or None after load.")

        # Apply transformations
        if matrix is not None:
            logger.debug(f"Applying transform matrix: {matrix}")
            structure_obj = transform_structure_with_matrix(structure_obj, matrix)
        else:
            logger.debug("Applying inertia transform")
            structure_obj = transform_structure_with_inertia(structure_obj)
        if (
            hasattr(structure_obj, "coordinates")
            and structure_obj.coordinates is not None
        ):
            logger.debug(
                f"Coordinates shape after transform: {structure_obj.coordinates.shape}"
            )
        else:
            logger.debug("Coordinates attribute not found or None after transform.")

        # Project structure orthographically
        logger.debug("Applying orthographic projection")
        structure_obj = project_structure_orthographically(
            structure_obj,
            canvas_width,
            canvas_height,
            maintain_aspect_ratio=True,
            center_projection=True,
        )
        if (
            hasattr(structure_obj, "coordinates")
            and structure_obj.coordinates is not None
        ):
            logger.debug(
                f"Coordinates shape after projection: {structure_obj.coordinates.shape}"
            )
        else:
            logger.debug("Coordinates attribute not found or None after projection.")

        styles_dict = None
        if style is not None:
            styles_dict = StyleParser(style).parse()

        scene = create_scene_from_structure(structure_obj, styles_dict)

        if annotations is not None:
            add_annotations_to_scene(annotations, scene)

        # Add position annotations based on level
        if show_positions != "none":
            position_style = None
            if styles_dict and "position_annotation" in styles_dict:
                position_style = styles_dict["position_annotation"]
            add_position_annotations_to_scene(scene, position_style, show_positions)

        # Log coordinate shape from scene before rendering
        if (
            hasattr(scene.structure, "coordinates")
            and scene.structure.coordinates is not None
        ):
            logger.debug(
                f"Final coordinates shape in scene before render: {scene.structure.coordinates.shape}"
            )
        else:
            logger.debug(
                "Coordinates attribute not found or None in scene structure before render."
            )

        renderer = SVGRenderer(
            scene, canvas_width, canvas_height, background_color=None
        )

        # Wrap rendering/saving in try...except
        try:
            if output is not None:
                renderer.save_svg(output)
            else:
                print(renderer.get_svg_string())
        except Exception as render_error:
            logger.error(f"Error during SVG rendering/saving: {render_error}")
            logger.exception(render_error)
            raise FlatProtError(f"Rendering failed: {render_error}") from render_error

        # Print success message using the success method (which uses INFO level)
        print_success_summary(structure, output, matrix, style, annotations, dssp)

        return 0

    except FlatProtError as e:
        logger.error(e.message)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1
