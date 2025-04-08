# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional, Annotated
from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter, Group, validators
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from flatprot.core import logger

verbosity_group = Group(
    "Verbosity",
    default_parameter=Parameter(negative=""),  # Disable "--no-" flags
    validator=validators.MutuallyExclusive(),  # Only one option is allowed to be selected.
)


@Parameter(name="*")
@dataclass
class CommonParameters:
    quiet: Annotated[
        bool,
        Parameter(group=verbosity_group, help="Suppress all output except errors."),
    ] = False
    verbose: Annotated[
        bool, Parameter(group=verbosity_group, help="Print additional information.")
    ] = False


def print_success_summary(
    structure_path: Path,
    output_path: Optional[Path],
    matrix_path: Optional[Path],
    style_path: Optional[Path],
    annotations_path: Optional[Path],
    dssp_path: Optional[Path],
) -> None:
    """
    Print a summary of the successful operation.

    This function logs information about the processed files and options
    used during the visualization generation.

    Args:
        structure_path: Path to the input structure file
        output_path: Path to the output SVG file, or None if printing to stdout
        matrix_path: Path to the custom transformation matrix file, or None if using default
        style_path: Path to the custom style file, or None if using default
        annotations_path: Path to the annotations file, or None if not using annotations
        dssp_path: Path to the DSSP file for secondary structure information, or None if using default
    """
    logger.info("[bold]Successfully processed structure:[/bold]")
    logger.info(f"  Structure file: {str(structure_path)}")
    logger.info(f"  Output file: {str(output_path) if output_path else 'stdout'}")
    logger.info(
        f"  Transformation: {'Custom matrix' if matrix_path else 'Inertia-based'}"
    )
    if matrix_path:
        logger.info(f"  Matrix file: {str(matrix_path)}")
    if style_path:
        logger.info(f"  Style file: {str(style_path)}")
    if annotations_path:
        logger.info(f"  Annotations file: {str(annotations_path)}")
    if dssp_path:
        logger.info(f"  DSSP file: {str(dssp_path)}")


# Create a theme for consistent styling
RICH_THEME = Theme(
    {
        "error": "bold red",
        "warning": "yellow",
        "info": "green",
        "debug": "blue",
    }
)


def __setup_logging(
    level: int = logging.WARNING,
    console: Optional[Console] = None,
) -> None:
    """Configure the logging system with Rich formatting.

    Args:
        level: Initial logging level (defaults to WARNING)
        console: Optional Rich console instance to use
    """
    # Create a console with our theme if not provided
    if console is None:
        console = Console(theme=RICH_THEME)

    # Clear existing handlers from the root logger
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    # Configure the root logger
    root.setLevel(level)

    # Use Rich's handler for console output
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        omit_repeated_times=False,
        show_path=False,
        enable_link_path=True,
        markup=True,  # Enable markup interpretation
        log_time_format="[%X]",
    )

    # Add the handler to the root logger
    root.addHandler(rich_handler)

    # Create a custom formatter class to handle level-specific formatting
    class LevelAwareFormatter(logging.Formatter):
        """Custom formatter that applies level-specific formatting."""

        def format(self, record: logging.LogRecord) -> str:
            """Format the log record with level-specific styling.

            Args:
                record: The log record to format

            Returns:
                The formatted log string
            """
            # Save the original message
            original_message = record.getMessage()

            # Apply level-specific formatting
            if record.levelno >= logging.ERROR:
                formatted_message = f"[error]{original_message}[/error]"
            elif record.levelno == logging.WARNING:
                formatted_message = f"[warning]{original_message}[/warning]"
            elif record.levelno == logging.INFO:
                formatted_message = f"[info]{original_message}[/info]"
            elif record.levelno <= logging.DEBUG:
                formatted_message = f"[debug]{original_message}[/debug]"
            else:
                formatted_message = original_message

            # Temporarily modify the message for formatting
            record.message = formatted_message

            # Apply the standard formatter
            result = super().format(record)

            # Restore the original message to avoid side effects
            record.message = original_message

            return result

    # Set the formatter on the Rich handler
    formatter = LevelAwareFormatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    rich_handler.setFormatter(formatter)


def set_logging_level(common: CommonParameters | None = None):
    if common and common.quiet:
        level = logging.ERROR
    elif common and common.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    __setup_logging(level)
