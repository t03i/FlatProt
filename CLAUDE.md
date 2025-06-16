# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlatProt is a Python package for creating simplified 2D protein visualizations, focusing on comparable representations for same-family proteins. The main CLI commands are:

- `flatprot project` - Creates 2D SVG projections from protein structures
- `flatprot align` - Aligns protein structures using rotation
- `flatprot overlay` - Creates overlay visualizations from multiple protein structures
- `flatprot split` - Extracts and aligns structural regions for comparative visualization

## Development Commands

### Package Management
- Install dependencies: `uv sync`
- Install with dev dependencies: `uv sync --group dev`
- Install with database builder dependencies: `uv sync --group db-builder`
- Use uv to run flatprot (new memory)

### Testing
- Only run python or pytest through uv
- Run all tests: `pytest`
- Run specific test module: `pytest tests/core/test_structure.py`
- Run tests with verbose output: `pytest -v`

### Building and Installation
- Build package: `uv build`
- Install locally: `uv tool install .`

### Documentation
- Build docs: `mkdocs build`
- Serve docs locally: `mkdocs serve`

[... rest of the existing content remains unchanged ...]
