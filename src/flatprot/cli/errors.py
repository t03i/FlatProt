# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

"""Error classes for the FlatProt CLI."""

import functools
import os
import sys
import traceback
from typing import Any, Callable, TypeVar, cast

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

from flatprot.core import FlatProtError

console = Console()

F = TypeVar("F", bound=Callable[..., Any])


class CLIError(FlatProtError):
    """Base class for CLI-specific errors."""

    def __init__(self, message: str):
        super().__init__(f"CLI error: {message}")


class CommandNotFoundError(CLIError):
    """Exception raised when a command is not found."""

    def __init__(self, command: str):
        message = f"Command not found: {command}"
        suggestion = "Run 'flatprot --help' to see available commands."
        super().__init__(f"{message}\n{suggestion}")


class InvalidArgumentError(CLIError):
    """Exception raised when an invalid argument is provided."""

    def __init__(self, argument: str, reason: str):
        message = f"Invalid argument: {argument}"
        suggestion = (
            f"Reason: {reason}\nRun 'flatprot {argument} --help' for more information."
        )
        super().__init__(f"{message}\n{suggestion}")


def error_handler(func: F) -> F:
    """Decorator to handle exceptions in CLI functions.

    This decorator catches exceptions raised by the decorated function and
    formats error messages using rich formatting. It also provides suggestions
    for fixing common issues.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FlatProtError as e:
            # Get stack trace information
            tb = sys.exc_info()[2]
            frame = traceback.extract_tb(tb)[-1]
            filename = os.path.basename(frame.filename)
            lineno = frame.lineno

            # Create rich text for the error message
            title = Text("FlatProt Error", style="bold red")
            error_type = Text(f"[{e.__class__.__name__}]", style="red")
            location = Text(f" at {filename}:{lineno}", style="dim")
            header = Text.assemble(title, " ", error_type, location)

            # Create panel with the error message
            panel = Panel(
                Text(e.message), title=header, border_style="red", padding=(1, 2)
            )

            # Print the error panel
            console.print(panel)
            return 1
        except Exception as e:
            # For unexpected exceptions, show more detailed traceback
            console.print("[bold red]Unexpected Error:[/bold red]", str(e))
            console.print(Traceback())
            return 1

    return cast(F, wrapper)
