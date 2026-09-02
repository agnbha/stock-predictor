"""News sentiment -- stretch, deliberately last (design section 3.4).

Headline sentiment is the feature most likely to leak (publication timestamps
are unreliable), most likely to be stale, and hardest to evaluate. Not started
until 3.1-3.3 are measurably working. Interface + null implementation only.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class NewsSentiment(Protocol):
    def sentiment(self, symbol: str, as_of_epoch: int, lookback_hours: int = 24) -> Optional[float]:
        """Net sentiment in [-1, 1] over headlines strictly before ``as_of_epoch``,
        or ``None`` if unknown. The strict cutoff is the leakage guard."""
        ...


class NullNewsSentiment:
    def sentiment(self, symbol: str, as_of_epoch: int, lookback_hours: int = 24) -> Optional[float]:  # noqa: ARG002
        return None


def get_default() -> NewsSentiment:
    return NullNewsSentiment()
