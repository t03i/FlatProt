class FlatProtError(Exception):
    """Base exception class for FlatProt CLI errors.

    This class is used as the base for all custom exceptions raised by the CLI.
    It provides a consistent interface for error handling and formatting.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
