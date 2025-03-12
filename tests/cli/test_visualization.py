# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FlatProt CLI visualization functionality."""

from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest
import numpy as np

from flatprot.cli.commands import generate_svg, save_svg
from flatprot.core import CoordinateManager, CoordinateType
from flatprot.core import (
    Structure,
    Chain,
    SecondaryStructureType,
    SecondaryStructure,
)


@pytest.fixture
def mock_structure():
    """Create a mock structure for testing."""
    structure = MagicMock(spec=Structure)

    # Create a mock chain
    chain = MagicMock(spec=Chain)
    chain.id = "A"

    # Create mock secondary structure elements
    helix = MagicMock(spec=SecondaryStructure)
    helix.start = 0
    helix.end = 2
    helix.secondary_structure_type = SecondaryStructureType.HELIX

    sheet = MagicMock(spec=SecondaryStructure)
    sheet.start = 3
    sheet.end = 5
    sheet.secondary_structure_type = SecondaryStructureType.SHEET

    # Add secondary structure elements to chain
    chain.secondary_structure = [helix, sheet]

    # Mock len() for chain
    chain.num_residues = 6

    # Add chain to structure
    structure.__iter__.return_value = [chain]

    # Mock coordinates
    structure.coordinates = np.array(
        [
            [1.0, 2.0, 3.0],  # 0
            [2.0, 3.0, 4.0],  # 1
            [3.0, 4.0, 5.0],  # 2
            [4.0, 5.0, 6.0],  # 3
            [5.0, 6.0, 7.0],  # 4
            [6.0, 7.0, 8.0],  # 5
        ]
    )

    return structure


@pytest.fixture
def mock_coordinate_manager():
    """Create a mock coordinate manager for testing."""
    cm = CoordinateManager()

    # Mock original coordinates
    original_coords = np.array(
        [
            [1.0, 2.0, 3.0],  # 0
            [2.0, 3.0, 4.0],  # 1
            [3.0, 4.0, 5.0],  # 2
            [4.0, 5.0, 6.0],  # 3
            [5.0, 6.0, 7.0],  # 4
            [6.0, 7.0, 8.0],  # 5
        ]
    )

    # Mock transformed coordinates
    transformed_coords = np.array(
        [
            [2.0, 3.0, 4.0],  # 0
            [3.0, 4.0, 5.0],  # 1
            [4.0, 5.0, 6.0],  # 2
            [5.0, 6.0, 7.0],  # 3
            [6.0, 7.0, 8.0],  # 4
            [7.0, 8.0, 9.0],  # 5
        ]
    )

    # Mock canvas coordinates
    canvas_coords = np.array(
        [
            [100.0, 200.0],  # 0
            [150.0, 250.0],  # 1
            [200.0, 300.0],  # 2
            [250.0, 350.0],  # 3
            [300.0, 400.0],  # 4
            [350.0, 450.0],  # 5
        ]
    )

    # Mock depth values
    depth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # Add all coordinates to the manager
    cm.add(0, 6, original_coords, CoordinateType.COORDINATES)
    cm.add(0, 6, transformed_coords, CoordinateType.TRANSFORMED)
    cm.add(0, 6, canvas_coords, CoordinateType.CANVAS)
    cm.add(0, 6, depth, CoordinateType.DEPTH)

    return cm


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

        # Verify annotations were parsed
        mock_annotation_parser.parse.assert_called_once_with(
            Path("mock_annotations.toml")
        )

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
