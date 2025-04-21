# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Parser for annotation files in TOML format with inline styles."""

import re
from pathlib import Path
from typing import List, Union, Dict, Any, Optional, Type, Callable, cast

import toml
from pydantic import ValidationError

from .errors import (
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    AnnotationError,
)

# Import the target Pydantic annotation and style models
from flatprot.scene.annotation import (
    PointAnnotation,
    LineAnnotation,
    AreaAnnotation,
    PointAnnotationStyle,
    LineAnnotationStyle,
    AreaAnnotationStyle,
    BaseAnnotationStyle,  # Import base style for type hint
)

# Import core types for coordinates/ranges
from flatprot.core import ResidueCoordinate, ResidueRange, ResidueRangeSet


# Helper function to parse "Chain:Index" string
def _parse_residue_coordinate(value: str, context: str) -> ResidueCoordinate:
    """Parses a string like 'A:123' into a ResidueCoordinate."""
    if not isinstance(value, str):
        raise MalformedAnnotationError(
            context, f"Expected string for coordinate, got {type(value).__name__}"
        )
    match = re.match(r"^([^:]+):(\d+)$", value)
    if not match:
        raise MalformedAnnotationError(
            context,
            f"Invalid coordinate format '{value}'. Expected 'ChainID:ResidueIndex'.",
        )
    chain_id = match.group(1)
    try:
        res_index = int(match.group(2))
    except ValueError:
        # Should not happen with regex, but defensive check
        raise MalformedAnnotationError(
            context, f"Invalid residue index in '{value}'. Must be an integer."
        )
    return ResidueCoordinate(chain_id=chain_id, residue_index=res_index)


# Helper function to parse "Chain:Start-End" string
def _parse_residue_range(value: str, context: str) -> ResidueRange:
    """Parses a string like 'A:100-150' into a ResidueRange."""
    if not isinstance(value, str):
        raise MalformedAnnotationError(
            context, f"Expected string for range, got {type(value).__name__}"
        )
    match = re.match(r"^([^:]+):(\d+)-(\d+)$", value)
    if not match:
        raise MalformedAnnotationError(
            context, f"Invalid range format '{value}'. Expected 'ChainID:Start-End'."
        )
    chain_id = match.group(1)
    try:
        start = int(match.group(2))
        end = int(match.group(3))
    except ValueError:
        # Should not happen with regex
        raise MalformedAnnotationError(
            context, f"Invalid start/end index in '{value}'. Must be integers."
        )
    if start > end:
        raise MalformedAnnotationError(
            context,
            f"Invalid range '{value}'. Start index ({start}) cannot be greater than end index ({end}).",
        )
    return ResidueRange(chain_id=chain_id, start=start, end=end)


# Type alias for the annotation objects
AnnotationObjectType = Union[PointAnnotation, LineAnnotation, AreaAnnotation]


class AnnotationParser:
    """Parses annotation files in TOML format with optional inline styles.

    Creates fully initialized PointAnnotation, LineAnnotation, or AreaAnnotation
    objects from the `flatprot.scene.annotations` module.
    """

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the parser with the path to the annotation file.

        Args:
            file_path: Path to the TOML file containing annotations.

        Raises:
            AnnotationFileNotFoundError: If the file does not exist.
        """
        self.file_path = Path(file_path)
        # Check file existence
        if not self.file_path.exists():
            raise AnnotationFileNotFoundError(str(self.file_path))

        # Map annotation type strings to their respective parsing methods
        self._parsers: Dict[
            str, Callable[[Dict[str, Any], str, str], AnnotationObjectType]
        ] = {
            "point": self._parse_point_annotation,
            "line": self._parse_line_annotation,
            "area": self._parse_area_annotation,
        }

    def _parse_inline_style(
        self,
        style_dict: Optional[Dict[str, Any]],
        StyleModel: Type[BaseAnnotationStyle],
        context: str,
    ) -> Optional[BaseAnnotationStyle]:
        """Parses the inline style dictionary using the provided Pydantic model."""
        if style_dict is None:
            return None
        if not isinstance(style_dict, dict):
            raise MalformedAnnotationError(
                context,
                f"'style' entry must be a table (dictionary), got {type(style_dict).__name__}.",
            )
        try:
            style_instance = StyleModel.model_validate(style_dict)
            return style_instance
        except ValidationError as e:
            error_details = e.errors()
            error_msgs = [f"  - {err['loc'][0]}: {err['msg']}" for err in error_details]
            raise MalformedAnnotationError(
                context, "Invalid style definition:\n" + "\n".join(error_msgs)
            ) from e
        except Exception as e:
            raise AnnotationError(f"{context}: Error creating style object: {e}") from e

    def _parse_point_annotation(
        self, anno_data: Dict[str, Any], anno_id: str, context: str
    ) -> PointAnnotation:
        """Parses a point annotation entry."""
        label = anno_data.get("label")
        index_str = anno_data.get("index")
        style_dict = anno_data.get("style")

        if index_str is None:
            raise MalformedAnnotationError(
                context, "Missing 'index' field for 'point' annotation."
            )

        target_coord = _parse_residue_coordinate(index_str, context)
        style_instance = self._parse_inline_style(
            style_dict, PointAnnotationStyle, context
        )

        return PointAnnotation(
            id=anno_id,
            target=target_coord,
            style=cast(Optional[PointAnnotationStyle], style_instance),
            label=label,
        )

    def _parse_line_annotation(
        self, anno_data: Dict[str, Any], anno_id: str, context: str
    ) -> LineAnnotation:
        """Parses a line annotation entry."""
        label = anno_data.get("label")
        indices_list = anno_data.get("indices")
        style_dict = anno_data.get("style")

        if not isinstance(indices_list, list) or len(indices_list) != 2:
            raise MalformedAnnotationError(
                context,
                "Field 'indices' for 'line' annotation must be a list of exactly two coordinate strings (e.g., ['A:10', 'A:20']).",
            )

        target_coords = [
            _parse_residue_coordinate(s, f"{context}, index {j + 1}")
            for j, s in enumerate(indices_list)
        ]
        style_instance = self._parse_inline_style(
            style_dict, LineAnnotationStyle, context
        )

        return LineAnnotation(
            id=anno_id,
            start_coordinate=target_coords[0],
            end_coordinate=target_coords[1],
            style=cast(
                Optional[LineAnnotationStyle], style_instance
            ),  # Cast for type checker
            label=label,
        )

    def _parse_area_annotation(
        self, anno_data: Dict[str, Any], anno_id: str, context: str
    ) -> AreaAnnotation:
        """Parses an area annotation entry."""
        label = anno_data.get("label")
        range_str = anno_data.get("range")
        style_dict = anno_data.get("style")
        # AreaAnnotation currently only supports range, not list of indices

        if range_str is None:
            raise MalformedAnnotationError(
                context, "Missing 'range' field for 'area' annotation."
            )

        target_range = _parse_residue_range(range_str, context)
        target_range_set = ResidueRangeSet([target_range])
        style_instance = self._parse_inline_style(
            style_dict, AreaAnnotationStyle, context
        )

        return AreaAnnotation(
            id=anno_id,
            residue_range_set=target_range_set,
            style=cast(
                Optional[AreaAnnotationStyle], style_instance
            ),  # Cast for type checker
            label=label,
        )

    def parse(self) -> List[AnnotationObjectType]:
        """Parse the annotation file and create annotation objects.

        Returns:
            List of parsed annotation objects (PointAnnotation, LineAnnotation, AreaAnnotation).

        Raises:
            MalformedAnnotationError: If the TOML file is malformed, missing required structure,
                                      contains invalid formats, or style validation fails.
            AnnotationError: For other general parsing issues.

        """
        try:
            # Parse TOML content
            raw_data = toml.load(self.file_path)
        except toml.TomlDecodeError as e:
            raise MalformedAnnotationError(
                f"File: {self.file_path}", f"Invalid TOML syntax: {str(e)}"
            ) from e
        except Exception as e:
            raise AnnotationError(
                f"Error loading TOML file {self.file_path}: {e}"
            ) from e

        if not isinstance(raw_data, dict) or "annotations" not in raw_data:
            raise MalformedAnnotationError(
                f"File: {self.file_path}", "Missing top-level 'annotations' list."
            )

        if not isinstance(raw_data["annotations"], list):
            raise MalformedAnnotationError(
                f"File: {self.file_path}", "'annotations' key must contain a list."
            )

        parsed_annotations: List[AnnotationObjectType] = []
        for i, anno_data in enumerate(raw_data["annotations"]):
            context = f"File: {self.file_path}, Annotation #{i + 1}"
            try:
                if not isinstance(anno_data, dict):
                    raise MalformedAnnotationError(
                        context, "Annotation entry must be a table (dictionary)."
                    )

                anno_type = anno_data.get("type")
                if not isinstance(anno_type, str):
                    raise MalformedAnnotationError(
                        context,
                        f"Missing or invalid 'type' field. Expected a string, got {type(anno_type).__name__}.",
                    )

                # Look up the parser function based on the type
                parser_func = self._parsers.get(anno_type)
                if parser_func is None:
                    raise MalformedAnnotationError(
                        context,
                        f"Unknown annotation 'type': '{anno_type}'. Must be one of {list(self._parsers.keys())}.",
                    )

                # Check for optional user-provided ID
                provided_id = anno_data.get("id")
                if provided_id is not None:
                    if isinstance(provided_id, str):
                        anno_id = provided_id
                    else:
                        raise MalformedAnnotationError(
                            context,
                            f"Optional 'id' field must be a string, got {type(provided_id).__name__}.",
                        )
                else:
                    # Generate ID if not provided
                    anno_id = f"annotation_{self.file_path.stem}_{i}_{anno_type}"

                # Call the specific parser method
                annotation_object = parser_func(anno_data, anno_id, context)
                parsed_annotations.append(annotation_object)

            except (MalformedAnnotationError, AnnotationError) as e:
                raise e
            except Exception as e:
                raise AnnotationError(
                    f"Unexpected error processing annotation in {context}: {e}"
                ) from e

        return parsed_annotations
