class FlatProtError(Exception):
    """Base exception class for FlatProt CLI errors.

    This class is used as the base for all custom exceptions raised by the CLI.
    It provides a consistent interface for error handling and formatting.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class CoordinateError(FlatProtError):
    """Error related to coordinate management."""

    def __init__(self, message: str):
        super().__init__(message)


class CoordinateCalculationError(CoordinateError):
    """Error raised when calculation of display coordinates fails (e.g., insufficient points)."""

    def __init__(self, message: str):
        super().__init__(message)
