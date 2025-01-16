# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class SecondaryStructureType(Enum):
    HELIX = "H"
    SHEET = "S"
    COIL = "O"
