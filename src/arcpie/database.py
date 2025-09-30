from __future__ import annotations

from pathlib import Path

from .featureclass import FeatureClass

class FileGeodatabase:
    def __init__(self, conn: str|Path) -> None:
        self.conn = conn