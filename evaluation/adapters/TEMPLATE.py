"""Copy this file when adding a new paper such as PIFu, GTA, or SIFU.

Only translate file names and metadata here.  Do not change metric definitions,
drop failed cases, align meshes, or rescale geometry in an adapter.
"""

from pathlib import Path
from typing import Any, Dict, List

from evaluation.adapters.base import ResultAdapter


class NewMethodAdapter(ResultAdapter):
    def convert(self, source: Path) -> List[Dict[str, Any]]:
        raise NotImplementedError("Map this method's outputs to evaluation.schema")

