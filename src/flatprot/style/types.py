# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class StyleType(Enum):
    """Type of style"""

    HELIX = "helix"
    SHEET = "sheet"
    COIL = "coil"
    CANVAS = "canvas"
    ANNOTATION = "annotation"
    ELEMENT = "element"
    AREA_ANNOTATION = "area_annotation"
    LINE_ANNOTATION = "line_annotation"
    POINT_ANNOTATION = "point_annotation"
