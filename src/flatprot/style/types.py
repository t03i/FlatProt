# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .structure_style import HelixStyle, SheetStyle, CoilStyle


class StyleType(Enum):
    """Type of style"""

    HELIX = "helix"
    SHEET = "sheet"
    COIL = "coil"
    POINT = "point"
    PAIR = "pair"
    GROUP = "group"


STYLE_MAP = {
    StyleType.HELIX: HelixStyle,
    StyleType.SHEET: SheetStyle,
    StyleType.COIL: CoilStyle,
}
