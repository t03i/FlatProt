#!/bin/bash
# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Find all Python files in the examples directory
find examples -maxdepth 1 -name "*.py" -print0 | while IFS= read -r -d $' ' file; do
    if [[ "$(basename "$file")" == "colab_setup.py" ]]; then
        echo "  Skipping setup script: $(basename "$file")"
        continue # Go to the next file
    fi

    echo "Converting ${file} to notebook..."
    uv run jupytext --to ipynb "${file}"
done

echo "Notebook conversion complete."
