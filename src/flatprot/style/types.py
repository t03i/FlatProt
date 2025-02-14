# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class StyleType(Enum):
    """Type of style"""

    HELIX = "helix"
    SHEET = "sheet"
    COIL = "coil"
    POINT = "point"
    PAIR = "pair"
    GROUP = "group"
    CANVAS = "canvas"
    ANNOTATION = "annotation"
    ELEMENT = "element"
