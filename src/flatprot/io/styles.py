# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Parser for style definitions in TOML format."""

from pathlib import Path
from typing import Dict, Any, Union, TypeVar
import toml
from pydantic import ValidationError
from pydantic_extra_types.color import Color

from ..style.manager import StyleManager
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

    # Change from REQUIRED_SECTIONS to KNOWN_SECTIONS
    KNOWN_SECTIONS = [
        "helix",
        "sheet",
        "coil",
        "point",
        "line",
        "area",
        "canvas",
        "annotation",
    ]

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

        Makes sure any provided sections have valid formats.
        All sections are optional - none are strictly required.
        """
        # Instead of checking for missing sections, just validate the ones that are present
        unknown_sections = [
            section for section in self.style_data if section not in self.KNOWN_SECTIONS
        ]

        if unknown_sections:
            # This is just a warning, not an error
            print(
                f"Warning: Unknown style sections found: {', '.join(unknown_sections)}"
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

    def _process_sections(self) -> Dict[str, Dict[str, Any]]:
        """Process all sections in the TOML file to prepare for StyleManager creation.

        Returns:
            Dict containing the processed style data with special handling for specific sections
        """
        processed_data = {}

        for section_name, section_data in self.style_data.items():
            # Skip unknown sections
            if section_name not in self.KNOWN_SECTIONS:
                continue

            # Copy data to avoid modifying the original
            processed_section = section_data.copy()

            # Special handling for point style (converting radius to line_width)
            if section_name == "point" and "radius" in processed_section:
                radius = processed_section.pop("radius")
                processed_section["line_width"] = radius * 2

            # Special handling for line style (converting stroke_width to line_width)
            if section_name == "line" and "stroke_width" in processed_section:
                processed_section["line_width"] = processed_section.pop("stroke_width")

            # Convert color strings to Color objects
            processed_section = self._convert_colors(processed_section)

            processed_data[section_name] = processed_section

        return processed_data

    def get_style_manager(self) -> StyleManager:
        """Create and return a StyleManager populated with styles from the TOML file.

        Returns:
            StyleManager instance populated with the parsed styles

        Raises:
            StyleValidationError: If any style section has invalid data
        """
        try:
            processed_data = self._process_sections()
            return StyleManager.from_dict(processed_data)
        except ValidationError as e:
            raise StyleValidationError(f"Invalid style definition: {e}")
        except Exception as e:
            raise StyleParsingError(f"Error creating style manager: {e}")

    def get_style_data(self) -> Dict[str, Any]:
        """Return the validated style data.

        Returns:
            Dict containing the parsed style data
        """
        return self.style_data
