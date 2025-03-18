# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import sys

from cyclopts import App

from flatprot.cli.commands import project_structure_svg, align_structure2_rotation
from flatprot import __version__


app = App(version=__version__)
app.default(project_structure_svg)
app.add_subcommand("align", align_structure2_rotation)


if __name__ == "__main__":
    sys.exit(app())
