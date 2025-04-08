# Copyright 2024 Rostlab.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for FlatProt."""

import logging
from flatprot import __app_name__


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


# Create the default logger instance
logger = getLogger()
