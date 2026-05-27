"""Abstract interface for email source loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.models import NormalizedEmail


class EmailLoader(ABC):
    """Abstract base class for all email sources."""

    @abstractmethod
    def load(self, num_samples: Optional[int] = None) -> List["NormalizedEmail"]:
        """Load emails from the source."""
        raise NotImplementedError

    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Return metadata about the source."""
        raise NotImplementedError
