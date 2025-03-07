# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Parser for annotation files in TOML format."""

from pathlib import Path
from typing import List, Optional, Union, Literal

import toml
from pydantic import BaseModel, ValidationError, field_validator, ConfigDict
from rich.console import Console

from flatprot.io.errors import (
    AnnotationFileNotFoundError,
    MalformedAnnotationError,
    InvalidReferenceError,
)
from flatprot.scene.annotations.point import PointAnnotation
from flatprot.scene.annotations.line import LineAnnotation
from flatprot.scene.annotations.area import AreaAnnotation
from flatprot.scene import Scene

console = Console()


# Pydantic models for validation


class PointAnnotationData(BaseModel):
    """Model for point annotation data in TOML files."""

    label: str
    type: Literal["point"]
    indices: int
    chain: str

    model_config = ConfigDict(frozen=True)


class LineAnnotationData(BaseModel):
    """Model for line/pair annotation data in TOML files."""

    label: str
    type: Literal["line", "pair"]
    indices: List[int]
    chain: str

    @field_validator("indices", mode="after")
    @classmethod
    def validate_indices(cls, v):
        if len(v) != 2:
            raise ValueError("Line/pair annotation must have exactly 2 indices")
        return v

    model_config = ConfigDict(frozen=True)


class AreaRange(BaseModel):
    """Model for area range data in TOML files."""

    start: int
    end: int

    @field_validator("end", mode="after")
    @classmethod
    def validate_end(cls, v, values):
        if "start" in values.data and v < values.data["start"]:
            raise ValueError("End must be greater than or equal to start")
        return v


class AreaAnnotationData(BaseModel):
    """Model for area annotation data in TOML files."""

    label: str
    type: Literal["area"]
    range: AreaRange
    chain: str

    model_config = ConfigDict(frozen=True)


class AnnotationsFile(BaseModel):
    """Model for the entire annotations file."""

    annotations: List[
        Union[PointAnnotationData, LineAnnotationData, AreaAnnotationData]
    ]

    model_config = ConfigDict(
        json_discriminator="type",
        json_schema_extra={
            "discriminator": {
                "propertyName": "type",
                "mapping": {
                    "point": "PointAnnotationData",
                    "line": "LineAnnotationData",
                    "pair": "LineAnnotationData",
                    "area": "AreaAnnotationData",
                },
            }
        },
    )


class AnnotationParser:
    """Parser for annotation files in TOML format."""

    def __init__(
        self,
        file_path: Path,
        scene: Optional[Scene] = None,
    ):
        """Initialize the parser with a file path and optional scene.

        Args:
            file_path: Path to the TOML file containing annotations
            scene: Optional scene object to map annotations to

        Raises:
            AnnotationFileNotFoundError: If the file does not exist
        """
        self.file_path = file_path
        self.scene = scene

        # Check file existence
        if not file_path.exists():
            raise AnnotationFileNotFoundError(str(file_path))

    def parse(self) -> List[Union[PointAnnotation, LineAnnotation, AreaAnnotation]]:
        """Parse the annotation file and create annotation objects.

        Returns:
            List of annotation objects

        Raises:
            MalformedAnnotationError: If the TOML file is malformed or missing required structure
            InvalidReferenceError: If an annotation references a nonexistent chain or residue
        """
        try:
            # Parse TOML content
            content = toml.load(self.file_path)

            # Validate using Pydantic
            annotations_file = AnnotationsFile.model_validate(content)

        except toml.TomlDecodeError as e:
            raise MalformedAnnotationError(
                str(self.file_path), f"Invalid TOML syntax: {str(e)}"
            )
        except ValidationError as e:
            raise MalformedAnnotationError(str(self.file_path), str(e))

        # If no scene provided, we just validated the TOML but can't create objects
        if not self.scene:
            return []

        # Create annotation objects
        annotation_objects = []
        for i, data in enumerate(annotations_file.annotations):
            # Validate scene references and create annotation objects
            if isinstance(data, PointAnnotationData):
                self._validate_scene_reference_point(data, i)
                annotation_objects.append(self._create_point_annotation(data))

            elif isinstance(data, LineAnnotationData):
                self._validate_scene_reference_line(data, i)
                annotation_objects.append(self._create_line_annotation(data))

            elif isinstance(data, AreaAnnotationData):
                self._validate_scene_reference_area(data, i)
                annotation_objects.append(self._create_area_annotation(data))

        return annotation_objects

    def _validate_scene_reference_point(
        self, data: PointAnnotationData, index: int
    ) -> None:
        """Validate scene references for a point annotation.

        Args:
            data: Annotation data
            index: Index of the annotation

        Raises:
            InvalidReferenceError: If the reference doesn't exist in the scene
        """
        elements = self.scene.get_elements_for_residue(data.chain, data.indices)
        if not elements:
            raise InvalidReferenceError(
                "point", "residue", f"{data.indices} in chain {data.chain}", index
            )

    def _validate_scene_reference_line(
        self, data: LineAnnotationData, index: int
    ) -> None:
        """Validate scene references for a line annotation.

        Args:
            data: Annotation data
            index: Index of the annotation

        Raises:
            InvalidReferenceError: If the references don't exist in the scene
        """
        for idx in data.indices:
            elements = self.scene.get_elements_for_residue(data.chain, idx)
            if not elements:
                raise InvalidReferenceError(
                    "line/pair", "residue", f"{idx} in chain {data.chain}", index
                )

    def _validate_scene_reference_area(
        self, data: AreaAnnotationData, index: int
    ) -> None:
        """Validate scene references for an area annotation.

        Args:
            data: Annotation data
            index: Index of the annotation

        Raises:
            InvalidReferenceError: If the reference range doesn't exist in the scene
        """
        elements = self.scene.get_elements_for_residue_range(
            data.chain, data.range.start, data.range.end
        )
        if not elements:
            raise InvalidReferenceError(
                "area",
                "residue range",
                f"{data.range.start}-{data.range.end} in chain {data.chain}",
                index,
            )

    def _create_point_annotation(self, data: PointAnnotationData) -> PointAnnotation:
        """Create a point annotation object.

        Args:
            data: Annotation data

        Returns:
            PointAnnotation object
        """
        targets = self.scene.get_elements_for_residue(data.chain, data.indices)

        return PointAnnotation(
            label=data.label,
            targets=targets,
            indices=[data.indices],
        )

    def _create_line_annotation(self, data: LineAnnotationData) -> LineAnnotation:
        """Create a line annotation object.

        Args:
            data: Annotation data

        Returns:
            LineAnnotation object
        """
        targets = []
        for idx in data.indices:
            elements = self.scene.get_elements_for_residue(data.chain, idx)
            if elements:
                targets.append(elements[0])  # Take the first element for each residue

        return LineAnnotation(
            label=data.label,
            targets=targets,
            indices=data.indices,
        )

    def _create_area_annotation(self, data: AreaAnnotationData) -> AreaAnnotation:
        """Create an area annotation object.

        Args:
            data: Annotation data

        Returns:
            AreaAnnotation object
        """
        targets = self.scene.get_elements_for_residue_range(
            data.chain, data.range.start, data.range.end
        )

        return AreaAnnotation(
            label=data.label,
            targets=targets,
        )
