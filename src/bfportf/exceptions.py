"""bfportf exceptions."""

from collections.abc import Iterable
from typing import Optional


class BfPortfError(Exception):
    """Base class for all bfportf errors."""


class DataValidationError(BfPortfError, ValueError):
    """Raised when the input data does not pass validation."""


class UnknownSymbolError(BfPortfError, KeyError):
    """Raised when a requested symbol is missing from the returns dataframe."""

    def __init__(self, missing: Iterable[str], available: Optional[Iterable[str]] = None) -> None:
        """Initialize the error.

        Args:
            missing (Iterable[str]): Symbols that were requested but not found.
            available (Iterable[str], optional): Symbols that are available. Defaults to None.
        """
        self.missing = tuple(missing)
        self.available = tuple(available) if available is not None else ()
        super().__init__(f"Symbols not found in returns: {', '.join(self.missing)}")

    def __str__(self) -> str:
        """Return a human readable message without the KeyError quoting."""
        return str(self.args[0]) if self.args else ""
