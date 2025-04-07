# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


"""Tests for SVG utilities in FlatProt."""

from pathlib import Path
from typing import Any
import pytest
from pytest_mock import MockerFixture

from flatprot.utils.svg import generate_svg, save_svg
from flatprot.core import CoordinateManager
from flatprot.core.structure import Structure, Chain
from flatprot.style import StyleManager
from flatprot.scene import Scene


@pytest.fixture
def mock_structure(mocker: MockerFixture) -> Structure:
    """Fixture providing a mock protein structure.

    Returns:
        Mock Structure object with a chain
    """
    mock_chain = mocker.MagicMock(spec=Chain)
    mock_chain.id = "A"

    # Configure mock residues with integer 'number' attributes
    mock_residue_start = mocker.MagicMock()
    mock_residue_start.number = 1
    mock_residue_end = mocker.MagicMock()
    mock_residue_end.number = 100
    mock_chain.residues = [mock_residue_start, mock_residue_end]

    mock_structure = mocker.MagicMock(spec=Structure)
    mock_structure.__iter__.return_value = [mock_chain]
    return mock_structure


@pytest.fixture
def mock_coordinate_manager(mocker: MockerFixture) -> CoordinateManager:
    """Fixture providing a mock coordinate manager.

    Returns:
        Mock CoordinateManager object
    """
    return mocker.MagicMock(spec=CoordinateManager)


@pytest.fixture
def mock_style_manager(mocker: MockerFixture) -> StyleManager:
    """Fixture providing a mock style manager.

    Returns:
        Mock StyleManager object
    """
    return mocker.MagicMock(spec=StyleManager)


@pytest.fixture
def mock_scene(mocker: MockerFixture) -> Scene:
    """Fixture providing a mock scene.

    Returns:
        Mock Scene object
    """
    return mocker.MagicMock(spec=Scene)


@pytest.fixture
def mock_canvas(mocker: MockerFixture) -> Any:
    """Fixture providing a mock canvas.

    Returns:
        Mock Canvas object
    """
    mock_canvas = mocker.MagicMock()
    mock_drawing = mocker.MagicMock()
    mock_drawing.as_svg.return_value = "<svg>test</svg>"
    mock_canvas.render.return_value = mock_drawing
    return mock_canvas


def test_generate_svg(
    mocker: MockerFixture,
    mock_structure: Structure,
    mock_coordinate_manager: CoordinateManager,
    mock_style_manager: StyleManager,
    mock_scene: Scene,
    mock_canvas: Any,
) -> None:
    """Test that SVG content is generated correctly from a structure.

    Args:
        mocker: Pytest mocker fixture
        mock_structure: Mock Structure object
        mock_coordinate_manager: Mock CoordinateManager object
        mock_style_manager: Mock StyleManager object
        mock_scene: Mock Scene object
        mock_canvas: Mock Canvas object
    """
    # Mock the Scene class and Canvas class
    mocker.patch("flatprot.utils.svg.Scene", return_value=mock_scene)
    mocker.patch("flatprot.utils.svg.Canvas", return_value=mock_canvas)

    # Mock the process_structure_chain function to return the scene object
    mock_process_chain = mocker.patch("flatprot.utils.svg.process_structure_chain")

    # Mock the process_annotations function
    mock_process_annotations = mocker.patch("flatprot.utils.svg.process_annotations")

    # Verify Canvas was created with the scene and style manager
    canvas_patch = mocker.patch("flatprot.utils.svg.Canvas", return_value=mock_canvas)

    # Call the function
    result = generate_svg(
        mock_structure, mock_coordinate_manager, mock_style_manager, None
    )

    canvas_patch.assert_called_once_with(mock_scene, mock_style_manager)

    # Verify process_structure_chain was called for each chain
    mock_process_chain.assert_called_once()

    # Verify process_annotations was not called (no annotations provided)
    mock_process_annotations.assert_not_called()

    # Verify render was called on the canvas
    mock_canvas.render.assert_called_once()

    # Verify that as_svg was called on the drawing
    mock_canvas.render().as_svg.assert_called_once()

    # Verify that the result is what we expect
    assert result == "<svg>test</svg>"


def test_generate_svg_with_annotations(
    mocker: MockerFixture,
    mock_structure: Structure,
    mock_coordinate_manager: CoordinateManager,
    mock_style_manager: StyleManager,
    mock_scene: Scene,
    mock_canvas: Any,
) -> None:
    """Test that SVG content is generated correctly with annotations.

    Args:
        mocker: Pytest mocker fixture
        mock_structure: Mock Structure object
        mock_coordinate_manager: Mock CoordinateManager object
        mock_style_manager: Mock StyleManager object
        mock_scene: Mock Scene object
        mock_canvas: Mock Canvas object
    """
    # Mock the Scene class and Canvas class
    mocker.patch("flatprot.utils.svg.Scene", return_value=mock_scene)
    canvas_patch = mocker.patch("flatprot.utils.svg.Canvas", return_value=mock_canvas)

    # Mock the process_structure_chain function
    mock_process_chain = mocker.patch("flatprot.utils.svg.process_structure_chain")

    # Mock the process_annotations function
    mock_process_annotations = mocker.patch("flatprot.utils.svg.process_annotations")

    # Create a test annotations path
    annotations_path = Path("/path/to/annotations.toml")

    # Call the function
    result = generate_svg(
        mock_structure, mock_coordinate_manager, mock_style_manager, annotations_path
    )

    # Verify process_structure_chain was called for each chain
    mock_process_chain.assert_called_once()

    # Verify process_annotations was called with the annotations path
    mock_process_annotations.assert_called_once_with(
        annotations_path, mock_scene, mock_style_manager
    )

    # Verify Canvas was created with the scene and style manager
    canvas_patch.assert_called_once_with(mock_scene, mock_style_manager)

    # Verify render was called on the canvas
    mock_canvas.render.assert_called_once()

    # Verify that as_svg was called on the drawing
    mock_canvas.render().as_svg.assert_called_once()

    # Verify that the result is what we expect
    assert result == "<svg>test</svg>"


def test_save_svg_success(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that SVG content is successfully saved to a file.

    Args:
        mocker: Pytest mocker fixture
        tmp_path: Pytest temporary directory fixture
    """
    # Mock logger.info instead of console.print
    mock_logger_info = mocker.patch("flatprot.utils.svg.logger.info")

    # Test data
    svg_content = "<svg>test</svg>"
    output_path = tmp_path / "subdir" / "test.svg"

    # Call the function
    save_svg(svg_content, output_path)

    # Verify the directory was created
    assert output_path.parent.exists()

    # Verify the file was created and contains the correct content
    assert output_path.exists()
    with open(output_path, "r") as f:
        assert f.read() == svg_content

    # Verify success message was logged
    mock_logger_info.assert_called_once()
    assert (
        f"[bold]SVG saved to {output_path}[/bold]" in mock_logger_info.call_args[0][0]
    )


def test_save_svg_existing_dir(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that SVG content is saved to an existing directory.

    Args:
        mocker: Pytest mocker fixture
        tmp_path: Pytest temporary directory fixture
    """
    # Mock logger.info instead of console.print
    mock_logger_info = mocker.patch("flatprot.utils.svg.logger.info")

    # Create the directory beforehand
    subdir = tmp_path / "existing_dir"
    subdir.mkdir()

    # Test data
    svg_content = "<svg>test</svg>"
    output_path = subdir / "test.svg"

    # Call the function
    save_svg(svg_content, output_path)

    # Verify the file was created and contains the correct content
    assert output_path.exists()
    with open(output_path, "r") as f:
        assert f.read() == svg_content

    # Verify success message was logged
    mock_logger_info.assert_called_once()
    assert (
        f"[bold]SVG saved to {output_path}[/bold]" in mock_logger_info.call_args[0][0]
    )


def test_save_svg_io_error(mocker: MockerFixture) -> None:
    """Test handling of IO errors when saving SVG content.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock logger.error instead of console.print
    mock_logger_error = mocker.patch("flatprot.utils.svg.logger.error")

    # Mock open to raise an error
    mock_open = mocker.patch("builtins.open")
    mock_open.side_effect = IOError("Mock IO error")

    # Mock os.makedirs
    mocker.patch("os.makedirs")

    # Test data
    svg_content = "<svg>test</svg>"
    output_path = Path("/path/to/test.svg")

    # Verify that the function raises an IOError
    with pytest.raises(IOError) as excinfo:
        save_svg(svg_content, output_path)

    # Verify error message
    assert "Failed to save SVG" in str(excinfo.value)

    # Verify error message was logged
    mock_logger_error.assert_called_once()
    assert "Error saving SVG" in mock_logger_error.call_args[0][0]


def test_save_svg_makedirs_error(mocker: MockerFixture) -> None:
    """Test handling of errors when creating directories.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock logger.error instead of console.print
    mock_logger_error = mocker.patch("flatprot.utils.svg.logger.error")

    # Mock os.makedirs to raise an error
    mock_makedirs = mocker.patch("os.makedirs")
    mock_makedirs.side_effect = PermissionError("Mock permission error")

    # Test data
    svg_content = "<svg>test</svg>"
    output_path = Path("/path/to/test.svg")

    # Verify that the function raises an IOError
    with pytest.raises(IOError) as excinfo:
        save_svg(svg_content, output_path)

    # Verify error message
    assert "Failed to save SVG" in str(excinfo.value)

    # Verify error message was logged
    mock_logger_error.assert_called_once()
    assert "Error saving SVG" in mock_logger_error.call_args[0][0]
