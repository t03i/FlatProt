# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .base import StructureComponent, SubStructureComponent


class Protein(StructureComponent):
    pass


class Helix(SubStructureComponent):
    pass


class Sheet(SubStructureComponent):
    pass


class Loop(SubStructureComponent):
    pass
