# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Parser for annotation files in TOML format."""

from pathlib import Path
from typing import Any

import toml
from rich.console import Console

console = Console()


class AnnotationError(Exception):
    """Base exception class for annotation parsing errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class AnnotationFileNotFoundError(AnnotationError):
    """Exception raised when an annotation file is not found."""

    def __init__(self, file_path: str):
        message = f"Annotation file not found: {file_path}"
        super().__init__(message)


class MalformedAnnotationError(AnnotationError):
    """Exception raised when an annotation file has an invalid format."""

    def __init__(self, file_path: str, details: str):
        message = f"Malformed annotation file: {file_path}\n{details}"
        super().__init__(message)


class MissingRequiredFieldError(AnnotationError):
    """Exception raised when a required field is missing from an annotation."""

    def __init__(self, annotation_type: str, field_name: str, annotation_index: int):
        message = f"Missing required field '{field_name}' for {annotation_type} annotation at index {annotation_index}"
        super().__init__(message)


class InvalidFieldTypeError(AnnotationError):
    """Exception raised when a field has an invalid type."""

    def __init__(
        self,
        annotation_type: str,
        field_name: str,
        expected_type: str,
        annotation_index: int,
    ):
        message = f"Invalid type for field '{field_name}' in {annotation_type} annotation at index {annotation_index}. Expected {expected_type}."
        super().__init__(message)


class AnnotationParser:
    """Parser for annotation files in TOML format."""

    def __init__(self, file_path: Path):
        """Initialize the parser with a file path.

        Args:
            file_path: Path to the TOML file containing annotations

        Raises:
            AnnotationFileNotFoundError: If the file does not exist
        """
        self.file_path = file_path

        # Check file existence
        if not file_path.exists():
            raise AnnotationFileNotFoundError(str(file_path))

    def parse(self) -> list[dict[str, Any]]:
        """Parse the annotation file and validate its structure.

        Returns:
            list of parsed annotation dictionaries

        Raises:
            MalformedAnnotationError: If the TOML file is malformed or missing required structure
            MissingRequiredFieldError: If an annotation is missing a required field
            InvalidFieldTypeError: If a field has an invalid type
        """
        try:
            # Parse TOML content
            content = toml.load(self.file_path)
        except toml.TomlDecodeError as e:
            raise MalformedAnnotationError(
                str(self.file_path), f"Invalid TOML syntax: {str(e)}"
            )

        # Check for 'annotations' list
        if "annotations" not in content:
            raise MalformedAnnotationError(
                str(self.file_path),
                "Missing 'annotations' list. The file should contain a list of annotations.",
            )

        annotations = content["annotations"]
        if not isinstance(annotations, list):
            raise MalformedAnnotationError(
                str(self.file_path), "'annotations' must be a list"
            )

        # Validate each annotation
        validated_annotations = []
        for i, annotation in enumerate(annotations):
            # Validate common required fields
            self._validate_common_fields(annotation, i)

            # Validate specific fields based on type
            annotation_type = annotation["type"]
            if annotation_type == "point":
                self._validate_point_annotation(annotation, i)
            elif annotation_type in ["pair", "line"]:
                self._validate_line_annotation(annotation, i)
            elif annotation_type == "area":
                self._validate_area_annotation(annotation, i)
            else:
                raise InvalidFieldTypeError(
                    "annotation", "type", "'point', 'pair', 'line', or 'area'", i
                )

            validated_annotations.append(annotation)

        return validated_annotations

    def _validate_common_fields(self, annotation: dict[str, Any], index: int) -> None:
        """Validate common fields required for all annotation types.

        Args:
            annotation: Annotation dictionary
            index: Index of the annotation in the list

        Raises:
            MissingRequiredFieldError: If a required field is missing
            InvalidFieldTypeError: If a field has an invalid type
        """
        # Check for required fields
        for field in ["label", "type"]:
            if field not in annotation:
                raise MissingRequiredFieldError("annotation", field, index)

        # Validate field types
        if not isinstance(annotation["label"], str):
            raise InvalidFieldTypeError("annotation", "label", "string", index)
        if not isinstance(annotation["type"], str):
            raise InvalidFieldTypeError("annotation", "type", "string", index)

    def _validate_point_annotation(
        self, annotation: dict[str, Any], index: int
    ) -> None:
        """Validate fields specific to point annotations.

        Args:
            annotation: Annotation dictionary
            index: Index of the annotation in the list

        Raises:
            MissingRequiredFieldError: If a required field is missing
            InvalidFieldTypeError: If a field has an invalid type
        """
        # Check for required fields
        for field in ["indices", "chain"]:
            if field not in annotation:
                raise MissingRequiredFieldError("point", field, index)

        # Validate field types
        if not isinstance(annotation["indices"], int):
            raise InvalidFieldTypeError("point", "indices", "integer", index)
        if not isinstance(annotation["chain"], str):
            raise InvalidFieldTypeError("point", "chain", "string", index)

    def _validate_line_annotation(self, annotation: dict[str, Any], index: int) -> None:
        """Validate fields specific to line/pair annotations.

        Args:
            annotation: Annotation dictionary
            index: Index of the annotation in the list

        Raises:
            MissingRequiredFieldError: If a required field is missing
            InvalidFieldTypeError: If a field has an invalid type
        """
        # Check for required fields
        for field in ["indices", "chain"]:
            if field not in annotation:
                raise MissingRequiredFieldError("line/pair", field, index)

        # Validate field types
        if not isinstance(annotation["indices"], list):
            raise InvalidFieldTypeError(
                "line/pair", "indices", "list of 2 integers", index
            )

        # Check indices list length and types
        indices = annotation["indices"]
        if len(indices) != 2:
            raise InvalidFieldTypeError(
                "line/pair", "indices", "list of 2 integers", index
            )

        for i, idx in enumerate(indices):
            if not isinstance(idx, int):
                raise InvalidFieldTypeError(
                    "line/pair", f"indices[{i}]", "integer", index
                )

        if not isinstance(annotation["chain"], str):
            raise InvalidFieldTypeError("line/pair", "chain", "string", index)

    def _validate_area_annotation(self, annotation: dict[str, Any], index: int) -> None:
        """Validate fields specific to area annotations.

        Args:
            annotation: Annotation dictionary
            index: Index of the annotation in the list

        Raises:
            MissingRequiredFieldError: If a required field is missing
            InvalidFieldTypeError: If a field has an invalid type
        """
        # Check for required fields
        for field in ["range", "chain"]:
            if field not in annotation:
                raise MissingRequiredFieldError("area", field, index)

        # Validate field types
        if not isinstance(annotation["range"], dict):
            raise InvalidFieldTypeError(
                "area", "range", "dictionary with start and end keys", index
            )

        # Check range dictionary
        range_dict = annotation["range"]
        for field in ["start", "end"]:
            if field not in range_dict:
                raise MissingRequiredFieldError("area range", field, index)

            if not isinstance(range_dict[field], int):
                raise InvalidFieldTypeError("area range", field, "integer", index)

        # Validate range values
        if range_dict["end"] < range_dict["start"]:
            raise InvalidFieldTypeError(
                "area range", "end", "integer greater than or equal to start", index
            )

        if not isinstance(annotation["chain"], str):
            raise InvalidFieldTypeError("area", "chain", "string", index)
