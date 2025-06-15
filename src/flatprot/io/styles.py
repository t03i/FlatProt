# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Parser for structure style definitions in TOML format."""

from pathlib import Path
from typing import Dict, Any, Union

import toml
from pydantic import ValidationError

from flatprot.scene.connection import ConnectionStyle
from flatprot.scene.structure import (
    BaseStructureStyle,
    HelixStyle,
    SheetStyle,
    CoilStyle,
)
from flatprot.scene.annotation import PositionAnnotationStyle

from .errors import (
    StyleParsingError,
    InvalidTomlError,
    StyleValidationError,
    StyleFileNotFoundError,
)


class StyleParser:
    """Parser for TOML style files focusing on structure elements and connections."""

    # Define known sections and their corresponding Pydantic models
    KNOWN_SECTIONS = {
        "helix": HelixStyle,
        "sheet": SheetStyle,
        "coil": CoilStyle,
        "connection": ConnectionStyle,
        "position_annotation": PositionAnnotationStyle,
    }

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the style parser.

        Args:
            file_path: Path to the TOML style file

        Raises:
            StyleFileNotFoundError: If the file doesn't exist
            InvalidTomlError: If the TOML is malformed
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise StyleFileNotFoundError(f"Style file not found: {self.file_path}")

        try:
            with open(self.file_path, "r") as f:
                self.raw_style_data = toml.load(f)
        except toml.TomlDecodeError as e:
            raise InvalidTomlError(f"Invalid TOML format: {e}")

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Checks for unknown top-level sections in the style file."""
        unknown_sections = [
            section
            for section in self.raw_style_data
            if section not in self.KNOWN_SECTIONS
        ]

        if unknown_sections:
            # This is just a warning, not an error
            print(
                f"Warning: Unknown style sections found and ignored: {', '.join(unknown_sections)}"
            )

    def parse(
        self,
    ) -> Dict[str, Union[BaseStructureStyle, ConnectionStyle, PositionAnnotationStyle]]:
        """Parses the known sections from the TOML file into Pydantic style objects.

        Returns:
            A dictionary mapping section names ('helix', 'sheet', 'coil', 'connection', 'position_annotation')
            to their corresponding Pydantic style model instances.

        Raises:
            StyleValidationError: If any style section has invalid data according to
                                  its Pydantic model.
            StyleParsingError: For other general parsing issues.
        """
        parsed_styles: Dict[
            str, Union[BaseStructureStyle, ConnectionStyle, PositionAnnotationStyle]
        ] = {}

        for section_name, StyleModelClass in self.KNOWN_SECTIONS.items():
            section_data = self.raw_style_data.get(section_name)

            if section_data is None:
                # Section not present in the file, skip it (will use default later)
                continue

            if not isinstance(section_data, dict):
                raise StyleValidationError(
                    f"Invalid format for section '{section_name}'. Expected a table (dictionary), got {type(section_data).__name__}."
                )

            try:
                # Pydantic handles validation and type conversion (including Color)
                style_instance = StyleModelClass(**section_data)
                parsed_styles[section_name] = style_instance
            except ValidationError as e:
                # Provide more context for validation errors
                error_details = e.errors()
                error_msgs = [
                    f"  - {err['loc'][0]}: {err['msg']}" for err in error_details
                ]
                raise StyleValidationError(
                    f"Invalid style definition in section '{section_name}':\\n"
                    + "\\n".join(error_msgs)
                ) from e
            except Exception as e:
                # Catch other potential errors during instantiation
                raise StyleParsingError(
                    f"Error processing style section '{section_name}': {e}"
                ) from e

        return parsed_styles

    def get_element_styles(
        self,
    ) -> Dict[str, Union[BaseStructureStyle, ConnectionStyle, PositionAnnotationStyle]]:
        """Parse and return the element styles defined in the TOML file.

        Returns:
            Dict mapping element type names ('helix', 'sheet', 'coil', 'connection', 'position_annotation')
            to their parsed Pydantic style objects. Sections not found in the TOML
            file will be omitted from the dictionary.

        Raises:
            StyleValidationError: If validation of any section fails.
            StyleParsingError: For general parsing issues.
        """
        try:
            return self.parse()
        except (StyleValidationError, StyleParsingError):
            # Re-raise exceptions from parse
            raise
        except Exception as e:
            # Catch unexpected errors during the overall process
            raise StyleParsingError(f"Failed to get element styles: {e}") from e

    def get_raw_data(self) -> Dict[str, Any]:
        """Return the raw, unprocessed style data loaded from the TOML file.

        Returns:
            Dict containing the raw parsed TOML data.
        """
        return self.raw_style_data
