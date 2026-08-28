"""Small interface that every paper-specific result adapter should follow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class ResultAdapter(ABC):
    """Translate model-specific outputs without recomputing model metrics."""

    @abstractmethod
    def convert(self, source: Path) -> List[Dict[str, Any]]:
        raise NotImplementedError

