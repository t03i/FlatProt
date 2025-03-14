# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for FlatProt."""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from flatprot import __app_name__


# Create a theme for consistent styling
RICH_THEME = Theme(
    {
        "error": "bold red",
        "warning": "yellow",
        "info": "green",
        "debug": "blue",
    }
)


def getLogger(name: str = __app_name__) -> logging.Logger:
    """Get a configured logger with the given name.

    Args:
        name: Logger name (defaults to "flatprot")

    Returns:
        Configured logging.Logger instance
    """
    # Get the logger
    logger = logging.getLogger(name)
    return logger


def setup_logging(
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


# Create the default logger instance
logger = getLogger()
