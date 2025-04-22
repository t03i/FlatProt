# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import sys

from cyclopts import App

from .project import project_structure_svg
from .align import align_structure_rotation
from flatprot import __version__


app = App(version=__version__)
app.command(
    project_structure_svg,
    "project",
)
app.command(
    align_structure_rotation,
    "align",
)


if __name__ == "__main__":
    sys.exit(app())
