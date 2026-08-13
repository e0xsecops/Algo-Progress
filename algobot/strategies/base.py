"""The strategy contract.

A strategy answers exactly one question: *given the bars up to and including
``t``, what position do I want to hold?* It never sizes the position, never
places an order and never sees the account — the risk manager and the broker
own those jobs. Keeping the split sharp is what lets the same strategy run
unchanged against a backtest or a live feed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd


class Strategy(ABC):
    """Base class for signal generators.

    Subclasses declare their tunable parameters in :attr:`params_schema` (name
    to default) and implement :meth:`generate_signals`.
    """

    name: ClassVar[str] = "base"
    params_schema: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.params_schema)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; "
                f"expected any of {sorted(self.params_schema)}"
            )
        self.params: dict[str, Any] = {**self.params_schema, **params}
        self.validate()

    def __getattr__(self, item: str) -> Any:
        # Parameters read naturally as attributes: self.fast, self.slow, ...
        try:
            return self.__dict__["params"][item]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {item!r}"
            ) from None

    def validate(self) -> None:
        """Hook for subclasses to reject nonsensical parameter combinations."""

    @property
    def warmup(self) -> int:
        """Bars needed before signals are trustworthy. Used for reporting."""
        return 0

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return the desired position per bar as -1 (short), 0 (flat) or +1 (long).

        The value at bar ``t`` must depend only on bars ``<= t``; the engine
        acts on it at the *next* bar, which is what makes the result tradable.
        """

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional: indicator columns to expose for plotting or debugging."""
        return pd.DataFrame(index=df.index)

    def describe(self) -> str:
        joined = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({joined})" if joined else self.name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.describe()}>"
