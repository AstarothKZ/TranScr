"""Measure Tk heartbeat during the real asynchronous Application capture path."""

import argparse
import json
import os
from pathlib import Path
from queue import Queue
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.environ.get("TRANSCR_BENCH_CODE_ROOT", str(Path(__file__).resolve().parents[1])))

import numpy as np
from PIL import Image

from core.application import Application
from core.config import AppConfig
from core.pipeline import RuntimeBundle
from modules.ocr.windows_ocr import WindowsOCRModule
from modules.ocr.ocr_worker import OCRWorker
from modules.translation.translator import Translator
from benchmarks.run_pipeline import ImageCapture, ReplayTranslator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("test/benchmarks/responsiveness.json")
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path.home() / "OneDrive" / "Desktop" / "исходное-5.png",
    )
    args = parser.parse_args()
    with TemporaryDirectory(prefix="transcr-responsiveness-") as folder:
        config = AppConfig(Path(folder))
        config.path = Path(folder) / "settings.json"
        with patch("core.application.AppConfig", return_value=config):
            app = Application()
        app.root.withdraw()
        image = np.array(Image.open(args.image).convert("RGB"))
        app.screen_capturer.close()
        app.screen_capturer = ImageCapture(image)
        app.screen_capturer.close = lambda: None
        app.selected_area = (0, 0, image.shape[1], image.shape[0])
        queue = Queue()
        engine = WindowsOCRModule()
        worker = OCRWorker(
            engine,
            lambda texts, elapsed, token: queue.put(("result", texts, elapsed, token)),
            lambda error, token: queue.put(("error", error, 0, token)),
        )
        translator = (
            ReplayTranslator(args.replay)
            if args.replay
            else Translator(provider="google")
        )
        app._runtime = RuntimeBundle(
            ("windows", "google", False, "argos", False), worker, queue, translator
        )
        app._active = True
        gaps = []
        last = started = perf_counter()
        result = {}
        timeout_id = None

        def finish(timed_out=False):
            if timeout_id is not None:
                app.root.after_cancel(timeout_id)
            result.update(
                elapsed_ms=round((perf_counter() - started) * 1000, 2),
                max_ui_gap_ms=round(max(gaps, default=0), 2),
                ticks=len(gaps),
                timed_out=timed_out,
                status=app.status_var.get(),
                replayed_translation=args.replay is not None,
            )
            app.shutdown()

        def heartbeat():
            nonlocal last
            now = perf_counter()
            gaps.append((now - last) * 1000)
            last = now
            if app._capture_id > 0 and not app._busy:
                finish()
            else:
                app.root.after(10, heartbeat)

        timeout_id = app.root.after(30000, lambda: finish(True))
        app.root.after(10, heartbeat)
        app.root.after(20, lambda: app._start_capture("responsiveness benchmark"))
        app.run()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["timed_out"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
