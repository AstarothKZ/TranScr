"""One UTF-8 application log per session, independent of working directory."""

import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class SessionFileHandler(logging.FileHandler):
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        path = directory / f"transcr_{stamp}_{uuid4().hex[:8]}.log"
        super().__init__(path, mode="x", encoding="utf-8")
        self.addFilter(
            lambda record: (
                record.name.startswith("transcr.") or record.levelno >= logging.WARNING
            )
        )
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
