# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

from cyclopts import App

from flatprot.cli.commands import main
from flatprot import __version__


app = App(version=__version__)
app.default(main)

if __name__ == "__main__":
    app()
