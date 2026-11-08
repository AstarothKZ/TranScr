"""Session logs must be separate and contain both UI and pipeline messages."""

import logging
import unittest
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory

from core.application import Application
from core.session_logging import SessionFileHandler


class SessionLoggingTests(unittest.TestCase):
    def test_two_launches_have_separate_logs_without_duplicate_records(self):
        loggers = [
            logging.getLogger(name)
            for name in ("transcr.application", "transcr.pipeline", "")
        ]
        saved = [
            (logger, list(logger.handlers), logger.level, logger.propagate)
            for logger in loggers
        ]
        with TemporaryDirectory() as directory:
            try:
                paths = []
                for label in ("first", "second"):
                    app = Application.__new__(Application)
                    app.project_root = Path(directory)
                    app._ui_log_queue = Queue()
                    app._setup_logger()
                    paths.append(Path(app._session_file_handler.baseFilename))
                    app._logger.info("application %s", label)
                    app._pipeline_logger.info("pipeline %s", label)
                    logging.getLogger("dependency").warning("dependency %s", label)
                    app._session_file_handler.flush()
                self.assertNotEqual(paths[0], paths[1])
                self.assertEqual(len(list((Path(directory) / "logs").glob("*.log"))), 2)
                for path, label in zip(paths, ("first", "second")):
                    text = path.read_text(encoding="utf-8")
                    for category in ("application", "pipeline", "dependency"):
                        self.assertEqual(text.count(f"{category} {label}"), 1)
                    self.assertNotIn("second" if label == "first" else "first", text)
            finally:
                for logger, handlers, level, propagate in saved:
                    for handler in list(logger.handlers):
                        if handler not in handlers:
                            logger.removeHandler(handler)
                            handler.close()
                    logger.handlers = handlers
                    logger.setLevel(level)
                    logger.propagate = propagate

    def test_unique_paths_even_when_created_immediately(self):
        with TemporaryDirectory() as directory:
            first = SessionFileHandler(Path(directory))
            second = SessionFileHandler(Path(directory))
            try:
                self.assertNotEqual(first.baseFilename, second.baseFilename)
            finally:
                first.close()
                second.close()
