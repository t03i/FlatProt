# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Styling utilities for FlatProt."""

from pathlib import Path
from typing import Optional


from flatprot.style import StyleManager
from flatprot.io import StyleParser
from flatprot.core import FlatProtError

from .logger import logger


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
            logger.info(f"Using custom styles from {style_path}")

            # Log applied styles for debugging
            style_data = style_parser.get_style_data()
            for section, properties in style_data.items():
                logger.debug(f"  Applied {section} style:")
                for prop, value in properties.items():
                    logger.debug(f"    {prop}: {value}")

        except FlatProtError as e:
            logger.warning(f"Could not load style file: {style_path} - {e}")
            style_manager = StyleManager.create_default()
            logger.warning("Falling back to default styles")
    else:
        style_manager = StyleManager.create_default()
        logger.info("Using default styles")

    return style_manager
