# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Tests for style utilities in FlatProt."""

from pathlib import Path
from typing import Dict, Any
import pytest
from pytest_mock import MockerFixture

from flatprot.utils.style import create_style_manager
from flatprot.style import StyleManager
from flatprot.io import StyleParser


@pytest.fixture
def mock_style_data() -> Dict[str, Dict[str, Any]]:
    """Fixture providing mock style data.

    Returns:
        Dictionary containing mock style sections and properties
    """
    return {
        "helix": {"stroke_color": "#FF0000", "fill_color": "#FFDDDD"},
        "sheet": {"stroke_color": "#0000FF", "fill_color": "#DDDDFF"},
        "canvas": {"width": 1000, "height": 800},
    }


@pytest.fixture
def mock_style_parser(
    mock_style_data: Dict[str, Dict[str, Any]], mocker: MockerFixture
):
    """Fixture providing a mocked StyleParser.

    Args:
        mock_style_data: The mock style data to return
        mocker: pytest-mock fixture

    Returns:
        Mocked StyleParser instance
    """
    mock_parser = mocker.MagicMock(spec=StyleParser)
    mock_parser.get_style_manager.return_value = mocker.MagicMock(spec=StyleManager)
    mock_parser.get_style_data.return_value = mock_style_data
    return mock_parser


@pytest.fixture
def mock_style_manager(mocker):
    """Fixture providing a mocked StyleManager.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mocked StyleManager instance
    """
    mock_manager = mocker.MagicMock(spec=StyleManager)
    return mock_manager


def test_create_style_manager_with_default(mocker: MockerFixture) -> None:
    """Test creating a style manager with default styles.

    Args:
        mocker: pytest-mock fixture
    """
    # Mock StyleManager.create_default
    mock_manager = mocker.MagicMock(spec=StyleManager)
    mocker.patch(
        "flatprot.style.StyleManager.create_default", return_value=mock_manager
    )

    # Mock console to avoid actual printing
    mock_console = mocker.patch("flatprot.utils.style.console")

    # Call the function with no style_path
    result = create_style_manager()

    # Verify the default style manager was created
    assert result is mock_manager
    assert result == mock_manager

    # Verify console output
    mock_console.print.assert_called_once_with("Using default styles")


def test_create_style_manager_with_custom_style(
    mocker: MockerFixture,
    mock_style_parser,
    mock_style_data: Dict[str, Dict[str, Any]],
) -> None:
    """Test creating a style manager with a custom style file.

    Args:
        mocker: pytest-mock fixture
        mock_style_parser: Mocked StyleParser instance
        mock_style_data: Mock style data
    """
    # Mock StyleParser initialization
    mock_style_parser_class = mocker.patch(
        "flatprot.utils.style.StyleParser", return_value=mock_style_parser
    )

    # Mock console to avoid actual printing
    mock_console = mocker.patch("flatprot.utils.style.console")

    # Mock Path.exists to avoid file system checks
    mocker.patch.object(Path, "exists", return_value=True)

    # Create a style path
    style_path = Path("/path/to/custom_style.toml")

    # Call the function with the style path
    _ = create_style_manager(style_path)

    # Verify the StyleParser was initialized with the correct path
    mock_style_parser_class.assert_called_once_with(file_path=style_path)

    # Verify get_style_manager was called
    mock_style_parser.get_style_manager.assert_called_once()

    # Verify console output
    mock_console.print.assert_any_call(f"Using custom styles from {style_path}")

    # Verify style sections were logged
    for section, properties in mock_style_data.items():
        mock_console.print.assert_any_call(f"  [blue]Applied {section} style:[/blue]")
        for prop, value in properties.items():
            mock_console.print.assert_any_call(f"    {prop}: {value}")


def test_create_style_manager_integration_default(mocker: MockerFixture) -> None:
    """Integration test for creating a style manager with default styles.

    Args:
        mocker: pytest-mock fixture
    """
    # Mock StyleManager.create_default to return a real StyleManager
    real_style_manager = StyleManager()
    mocker.patch(
        "flatprot.style.StyleManager.create_default", return_value=real_style_manager
    )

    # Mock console to avoid actual printing
    mocker.patch("flatprot.utils.style.console")

    # Call the function with no style_path
    result = create_style_manager()

    # Verify we got back a real StyleManager
    assert isinstance(result, StyleManager)
    assert result is real_style_manager
