# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FlatProt CLI visualization functionality."""

from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

from flatprot.cli.commands import generate_svg, save_svg
from flatprot.scene.annotations import PointAnnotation, LineAnnotation, AreaAnnotation


def test_generate_svg_basic(mock_structure, mock_coordinate_manager):
    """Test generating SVG with default styles and no annotations."""
    with (
        patch("flatprot.cli.commands.StyleManager") as MockStyleManager,
        patch("flatprot.cli.commands.Canvas") as MockCanvas,
    ):
        # Setup mocks
        mock_style_manager = MagicMock()
        MockStyleManager.create_default.return_value = mock_style_manager

        # Mock the style parameters with proper return values
        mock_style = MagicMock()
        mock_style_manager.get_style.return_value = mock_style
        # Configure the __mul__ method to return a float
        mock_style.line_width.__mul__.return_value = 4.0
        # Set min_sheet_length to an int to avoid MagicMock comparison issues
        mock_style.min_sheet_length = 3

        mock_canvas = MagicMock()
        MockCanvas.return_value = mock_canvas

        mock_drawing = MagicMock()
        mock_canvas.render.return_value = mock_drawing
        mock_drawing.as_svg.return_value = "<svg>Test SVG</svg>"

        # Call function
        result = generate_svg(mock_structure, mock_coordinate_manager)

        # Verify default style manager was created
        MockStyleManager.create_default.assert_called_once()

        # Verify Canvas was created - use ANY to match the Scene object
        MockCanvas.assert_called_once_with(ANY, mock_style_manager)

        # Verify render was called and SVG was generated
        mock_canvas.render.assert_called_once()
        mock_drawing.as_svg.assert_called_once()

        # Verify result
        assert result == "<svg>Test SVG</svg>"


def test_generate_svg_with_style(mock_structure, mock_coordinate_manager):
    """Test generating SVG with custom style file."""
    with (
        patch("flatprot.cli.commands.StyleParser") as MockStyleParser,
        patch("flatprot.cli.commands.Canvas") as MockCanvas,
    ):
        # Setup mocks
        mock_style_parser = MagicMock()
        MockStyleParser.return_value = mock_style_parser

        mock_style_manager = MagicMock()
        mock_style_parser.get_styles.return_value = mock_style_manager

        # Mock the style parameters with proper return values
        mock_style = MagicMock()
        mock_style_manager.get_style.return_value = mock_style
        # Configure the __mul__ method to return a float
        mock_style.line_width.__mul__.return_value = 4.0
        # Set min_sheet_length to an int to avoid MagicMock comparison issues
        mock_style.min_sheet_length = 3

        mock_canvas = MagicMock()
        MockCanvas.return_value = mock_canvas

        mock_drawing = MagicMock()
        mock_canvas.render.return_value = mock_drawing
        mock_drawing.as_svg.return_value = "<svg>Custom Style SVG</svg>"

        # Call function with mock style path
        result = generate_svg(
            mock_structure, mock_coordinate_manager, style_path=Path("mock_style.toml")
        )

        # Verify style was parsed
        MockStyleParser.assert_called_once_with(file_path=Path("mock_style.toml"))
        mock_style_parser.get_styles.assert_called_once()

        # Verify Canvas was created with style manager - use ANY to match the Scene object
        MockCanvas.assert_called_once_with(ANY, mock_style_manager)

        # Verify render was called and SVG was generated
        mock_canvas.render.assert_called_once()
        mock_drawing.as_svg.assert_called_once()

        # Verify result
        assert result == "<svg>Custom Style SVG</svg>"


def test_generate_svg_with_annotations(mock_structure, mock_coordinate_manager):
    """Test generating SVG with annotations."""
    with (
        patch("flatprot.cli.commands.StyleManager") as MockStyleManager,
        patch("flatprot.cli.commands.AnnotationParser") as MockAnnotationParser,
        patch("flatprot.cli.commands.Canvas") as MockCanvas,
    ):
        # Setup mocks
        mock_style_manager = MagicMock()
        MockStyleManager.create_default.return_value = mock_style_manager

        # Mock the style parameters with proper return values
        mock_style = MagicMock()
        mock_style_manager.get_style.return_value = mock_style
        # Configure the __mul__ method to return a float
        mock_style.line_width.__mul__.return_value = 4.0
        # Set min_sheet_length to an int to avoid MagicMock comparison issues
        mock_style.min_sheet_length = 3

        mock_annotation_parser = MagicMock()
        MockAnnotationParser.return_value = mock_annotation_parser

        mock_annotation1 = MagicMock()
        mock_annotation2 = MagicMock()
        mock_annotation_parser.parse.return_value = [mock_annotation1, mock_annotation2]

        mock_canvas = MagicMock()
        MockCanvas.return_value = mock_canvas

        mock_drawing = MagicMock()
        mock_canvas.render.return_value = mock_drawing
        mock_drawing.as_svg.return_value = "<svg>Annotated SVG</svg>"

        # Call function with mock annotations path
        result = generate_svg(
            mock_structure,
            mock_coordinate_manager,
            annotations_path=Path("mock_annotations.toml"),
        )

        # Verify annotations were parsed with scene
        MockAnnotationParser.assert_called_once_with(
            file_path=Path("mock_annotations.toml"), scene=ANY
        )
        mock_annotation_parser.parse.assert_called_once()

        # Verify annotations were applied
        mock_annotation1.apply.assert_called_once()
        mock_annotation2.apply.assert_called_once()

        # Verify Canvas was created - use ANY to match the Scene object
        MockCanvas.assert_called_once_with(ANY, mock_style_manager)

        # Verify render was called and SVG was generated
        mock_canvas.render.assert_called_once()
        mock_drawing.as_svg.assert_called_once()

        # Verify result
        assert result == "<svg>Annotated SVG</svg>"


def test_save_svg_creates_directory():
    """Test saving SVG creates directory if needed."""
    with (
        patch("os.makedirs") as mock_makedirs,
        patch("builtins.open", new_callable=MagicMock) as mock_open,
        patch("flatprot.cli.commands.console") as mock_console,
    ):
        # Setup mock file
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Call function with path that has a directory
        output_path = Path("nonexistent_dir/test.svg")
        save_svg("<svg>Test</svg>", output_path)

        # Verify directory creation was attempted (with the correct path)
        mock_makedirs.assert_called_once_with(output_path.parent, exist_ok=True)

        # Verify file was opened and written to
        mock_open.assert_called_once_with(output_path, "w")
        mock_file.write.assert_called_once_with("<svg>Test</svg>")

        # Verify success message
        mock_console.print.assert_called_once()


def test_save_svg_handles_error():
    """Test saving SVG handles IO errors."""
    with (
        patch("os.makedirs") as mock_makedirs,  # noqa: F841
        patch("builtins.open", create=True) as mock_open,
        patch("flatprot.cli.commands.console") as mock_console,
    ):
        # Setup mock to raise an error
        mock_open.side_effect = IOError("Test error")

        # Call function and check for exception
        with pytest.raises(IOError):
            save_svg("<svg>Test</svg>", Path("test.svg"))

        # Verify error message was printed
        assert mock_console.print.called


def test_main_generates_and_saves_svg():
    """Test that the main function generates and saves an SVG."""
    with (
        patch("flatprot.cli.commands.validate_structure_file"),
        patch("flatprot.cli.commands.GemmiStructureParser") as MockParser,
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_get_cm,
        patch("flatprot.cli.commands.generate_svg") as mock_generate_svg,
        patch("flatprot.cli.commands.save_svg") as mock_save_svg,
        patch("flatprot.cli.commands.console"),
    ):
        # Setup mocks
        mock_structure = MagicMock()
        MockParser.return_value.parse_structure.return_value = mock_structure

        mock_coordinate_manager = MagicMock()
        mock_get_cm.return_value = mock_coordinate_manager

        mock_svg_content = "<svg>Test SVG</svg>"
        mock_generate_svg.return_value = mock_svg_content

        # Create file paths
        structure_file = Path("test.pdb")
        output_file = Path("output.svg")

        # Call main function
        from flatprot.cli.commands import main

        result = main(structure_file, output_file)

        # Verify functions were called with correct parameters
        MockParser.return_value.parse_structure.assert_called_once_with(structure_file)
        mock_get_cm.assert_called_once_with(mock_structure, None)
        mock_generate_svg.assert_called_once_with(
            mock_structure, mock_coordinate_manager, None, None
        )
        mock_save_svg.assert_called_once_with(mock_svg_content, output_file)

        # Verify success code
        assert result == 0


def test_main_with_all_options():
    """Test main function with all optional parameters."""
    with (
        patch("flatprot.cli.commands.validate_structure_file"),
        patch("flatprot.cli.commands.GemmiStructureParser") as MockParser,
        patch("flatprot.cli.commands.get_coordinate_manager") as mock_get_cm,
        patch("flatprot.cli.commands.generate_svg") as mock_generate_svg,
        patch("flatprot.cli.commands.save_svg") as mock_save_svg,
        patch("flatprot.cli.commands.console"),
        patch("pathlib.Path.exists") as mock_exists,
    ):
        # Setup mocks
        mock_structure = MagicMock()
        MockParser.return_value.parse_structure.return_value = mock_structure

        mock_coordinate_manager = MagicMock()
        mock_get_cm.return_value = mock_coordinate_manager

        mock_svg_content = "<svg>Test SVG with all options</svg>"
        mock_generate_svg.return_value = mock_svg_content

        # Set up file existence check
        mock_exists.return_value = True

        # Create file paths
        structure_file = Path("test.pdb")
        output_file = Path("output.svg")
        matrix_file = Path("transform.npy")
        annotations_file = Path("annotations.toml")
        style_file = Path("style.toml")

        # Call main function with all options
        from flatprot.cli.commands import main

        result = main(
            structure_file,
            output_file,
            matrix=matrix_file,
            annotations=annotations_file,
            style=style_file,
        )

        # Verify functions were called with correct parameters
        MockParser.return_value.parse_structure.assert_called_once_with(structure_file)
        mock_get_cm.assert_called_once_with(mock_structure, matrix_file)
        mock_generate_svg.assert_called_once_with(
            mock_structure, mock_coordinate_manager, annotations_file, style_file
        )
        mock_save_svg.assert_called_once_with(mock_svg_content, output_file)

        # Verify success code
        assert result == 0


def test_main_handles_errors():
    """Test main function handles errors gracefully."""
    with (
        patch("flatprot.cli.commands.validate_structure_file"),
        patch("flatprot.cli.commands.GemmiStructureParser") as MockParser,
        patch("flatprot.cli.commands.console") as mock_console,
    ):
        # Setup parser to raise an exception
        MockParser.return_value.parse_structure.side_effect = Exception("Test error")

        # Call main function
        from flatprot.cli.commands import main

        result = main(Path("test.pdb"), Path("output.svg"))

        # Verify error handling
        mock_console.print.assert_called()

        # Verify error code
        assert result == 1


@pytest.mark.parametrize(
    "annotation_type,expected_class",
    [
        ("point", PointAnnotation),
        ("line", LineAnnotation),
        ("area", AreaAnnotation),
    ],
)
def test_annotation_integration(
    mock_structure,
    mock_coordinate_manager,
    temp_annotation_file,
    annotation_type,
    expected_class,
):
    """Test integration with different annotation types."""
    with (
        patch("flatprot.cli.commands.StyleManager") as MockStyleManager,
        patch("flatprot.cli.commands.AnnotationParser") as MockAnnotationParser,
        patch("flatprot.cli.commands.Canvas") as MockCanvas,
        patch("flatprot.io.annotations.PointAnnotation") as MockPointAnnotation,
        patch("flatprot.io.annotations.LineAnnotation") as MockLineAnnotation,
        patch("flatprot.io.annotations.AreaAnnotation") as MockAreaAnnotation,
    ):
        # Setup mocks
        mock_style_manager = MagicMock()
        MockStyleManager.create_default.return_value = mock_style_manager

        # Create annotations based on the type
        mock_annotations = []

        # Setup the actual annotation parser with pass-through to the real implementation
        mock_annotation_parser = MagicMock()
        MockAnnotationParser.return_value = mock_annotation_parser

        # Create a mock annotation of the expected type
        mock_annotation = MagicMock(spec=expected_class)

        # Return only this type of annotation for the test
        if annotation_type == "point":
            MockPointAnnotation.return_value = mock_annotation
            mock_annotations = [mock_annotation]
        elif annotation_type == "line":
            MockLineAnnotation.return_value = mock_annotation
            mock_annotations = [mock_annotation]
        elif annotation_type == "area":
            MockAreaAnnotation.return_value = mock_annotation
            mock_annotations = [mock_annotation]

        mock_annotation_parser.parse.return_value = mock_annotations

        mock_canvas = MagicMock()
        MockCanvas.return_value = mock_canvas

        mock_drawing = MagicMock()
        mock_canvas.render.return_value = mock_drawing
        mock_drawing.as_svg.return_value = f"<svg>{annotation_type} annotation</svg>"

        # Call generate_svg with our temp annotation file
        result = generate_svg(
            mock_structure,
            mock_coordinate_manager,
            annotations_path=temp_annotation_file,
        )

        # Verify the annotation parser was initialized with our file
        MockAnnotationParser.assert_called_once_with(
            file_path=temp_annotation_file, scene=ANY
        )

        # Verify the annotation was applied
        mock_annotation.apply.assert_called_once()

        # Verify the result contains our annotation type
        assert f"<svg>{annotation_type} annotation</svg>" == result


def test_end_to_end_with_annotations(
    mock_structure, mock_coordinate_manager, temp_annotation_file, temp_style_file
):
    """Test an end-to-end scenario with actual annotations, not mocked."""
    # Only patch the Canvas to avoid actual drawing, but let the annotation parser run for real
    with patch("flatprot.cli.commands.Canvas") as MockCanvas:
        mock_canvas = MagicMock()
        MockCanvas.return_value = mock_canvas

        mock_drawing = MagicMock()
        mock_canvas.render.return_value = mock_drawing
        mock_drawing.as_svg.return_value = "<svg>End-to-end test</svg>"

        # Call generate_svg with actual annotation and style files
        result = generate_svg(
            mock_structure,
            mock_coordinate_manager,
            annotations_path=temp_annotation_file,
            style_path=temp_style_file,
        )

        # Verify Canvas was created and rendering occurred
        MockCanvas.assert_called_once()
        mock_canvas.render.assert_called_once()

        # Verify we got an SVG result
        assert result == "<svg>End-to-end test</svg>"
