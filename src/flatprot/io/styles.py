# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Parser for style definitions in TOML format."""

from pathlib import Path
from typing import Dict, Any, Union, TypeVar
import toml
from pydantic import ValidationError
from pydantic_extra_types.color import Color

from ..style.types import StyleType
from ..style.structure import HelixStyle, SheetStyle, ElementStyle, CoilStyle
from ..style.annotation import (
    AreaAnnotationStyle,
)
from ..style.base import Style


StyleT = TypeVar("StyleT", bound=Style)


class StyleParsingError(Exception):
    """Base error for style parsing issues."""

    pass


class InvalidTomlError(StyleParsingError):
    """Error for malformed TOML files."""

    pass


class StyleValidationError(StyleParsingError):
    """Error for invalid style field types or values."""

    pass


class StyleParser:
    """Parser for TOML style files."""

    REQUIRED_SECTIONS = ["helix", "sheet", "point", "line", "area"]

    # Mapping of TOML section names to style classes
    STYLE_MAPPING = {
        "helix": (StyleType.HELIX, HelixStyle),
        "sheet": (StyleType.SHEET, SheetStyle),
        "coil": (StyleType.COIL, CoilStyle),
        "point": (StyleType.POINT, ElementStyle),
        "line": (StyleType.ELEMENT, ElementStyle),
        "area": (StyleType.AREA_ANNOTATION, AreaAnnotationStyle),
    }

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the style parser.

        Args:
            file_path: Path to the TOML style file

        Raises:
            FileNotFoundError: If the file doesn't exist
            InvalidTomlError: If the TOML is malformed
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Style file not found: {self.file_path}")

        try:
            with open(self.file_path, "r") as f:
                self.style_data = toml.load(f)
        except toml.TomlDecodeError as e:
            raise InvalidTomlError(f"Invalid TOML format: {e}")

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validates the basic structure of the style file.

        Raises:
            StyleValidationError: If required sections are missing
        """
        missing_sections = [
            section
            for section in self.REQUIRED_SECTIONS
            if section not in self.style_data
        ]

        if missing_sections:
            raise StyleValidationError(
                f"Missing required style sections: {', '.join(missing_sections)}"
            )

    def _convert_colors(self, section_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string color values to Color objects.

        Args:
            section_data: Dictionary of style properties

        Returns:
            Dict with color strings converted to Color objects
        """
        converted_data = section_data.copy()

        # Look for color fields and convert them
        for key, value in section_data.items():
            if isinstance(value, str) and ("color" in key.lower()):
                try:
                    converted_data[key] = Color(value)
                except ValueError as e:
                    raise StyleValidationError(f"Invalid color format for '{key}': {e}")

        return converted_data

    def get_style_data(self) -> Dict[str, Any]:
        """Return the validated style data.

        Returns:
            Dict containing the parsed style data
        """
        return self.style_data

    def create_style(self, section_name: str) -> Style:
        """Create a style object from a section in the TOML file.

        Args:
            section_name: Name of the section in the TOML file

        Returns:
            Style object instance

        Raises:
            StyleValidationError: If the section data is invalid
            KeyError: If the section name is not recognized
        """
        if section_name not in self.STYLE_MAPPING:
            raise KeyError(f"Unknown style section: {section_name}")

        _, style_class = self.STYLE_MAPPING[section_name]
        section_data = self.style_data.get(section_name, {})

        # Copy data to avoid modifying the original
        section_data = section_data.copy()

        # Special handling for point style (converting radius to line_width)
        if section_name == "point" and "radius" in section_data:
            radius = section_data.pop("radius")
            section_data["line_width"] = radius * 2

        # Special handling for line style (converting stroke_width to line_width)
        if section_name == "line" and "stroke_width" in section_data:
            section_data["line_width"] = section_data.pop("stroke_width")

        # Convert color strings to Color objects
        converted_data = self._convert_colors(section_data)

        try:
            # Create the style object using Pydantic's model_validate
            return style_class.model_validate(converted_data)
        except ValidationError as e:
            raise StyleValidationError(f"Invalid style section '{section_name}': {e}")

    def get_styles(self) -> Dict[StyleType, Style]:
        """Get all style objects from the parsed data.

        Returns:
            Dictionary mapping style types to their corresponding style objects

        Raises:
            StyleValidationError: If any style section is invalid
        """
        styles = {}

        for section_name, (style_type, _) in self.STYLE_MAPPING.items():
            try:
                styles[style_type] = self.create_style(section_name)
            except Exception as e:
                raise StyleValidationError(f"Error creating {section_name} style: {e}")

        return styles
