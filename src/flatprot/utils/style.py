# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Styling utilities for FlatProt."""

from pathlib import Path
from typing import Optional

from rich.console import Console

from flatprot.style import StyleManager
from flatprot.io import StyleParser
from flatprot.core import FlatProtError

console = Console()


def create_style_manager(style_path: Optional[Path] = None) -> StyleManager:
    """Create a style manager from file or default.

    Args:
        style_path: Optional path to style file

    Returns:
        StyleManager with loaded styles
    """
    if style_path:
        try:
            style_parser = StyleParser(file_path=style_path)
            style_manager = style_parser.get_style_manager()
            console.print(f"Using custom styles from {style_path}")

            # Log applied styles for debugging
            style_data = style_parser.get_style_data()
            for section, properties in style_data.items():
                console.print(f"  [blue]Applied {section} style:[/blue]")
                for prop, value in properties.items():
                    console.print(f"    {prop}: {value}")

        except FlatProtError as e:
            console.print(
                f"[yellow]Warning: Could not load style file: {style_path} - {e}[/yellow]"
            )
            style_manager = StyleManager.create_default()
            console.print("Falling back to default styles")
    else:
        style_manager = StyleManager.create_default()
        console.print("Using default styles")

    return style_manager
