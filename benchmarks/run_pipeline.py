"""Repeatable image benchmarks using the application's real processing/rendering path.

Run each suite before/after a change. Results go to the ignored test/ directory;
the source images, their scale, and existing user preferences are never changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(
    0,
    os.environ.get("TRANSCR_BENCH_CODE_ROOT", str(Path(__file__).resolve().parents[1])),
)

import httpx
import numpy as np
from PIL import Image, ImageTk

from core.pipeline import FrameCache, RuntimeBundle, process_capture
from modules.ocr.ocr_worker import OCRWorker
from modules.overlay.screen_overlay import ScreenOverlay
from modules.translation.translator import Translator
from modules.translation.catalog import PROVIDER_LABELS


class ImageCapture:
    def __init__(self, image: np.ndarray):
        self.image = image

    def capture_region(self, area, *, save_debug=False):
        return self.image, None


class ReplayTranslator:
    """Hold translation content fixed when comparing CPU/rendering changes."""

    def __init__(self, path):
        record = json.loads(path.read_text(encoding="utf-8"))
        self.values = {
            item["region"]["text"].strip(): item["text"]
            for item in record["translations"]
        }

    def translate_many(self, texts, cancel=None):
        return {text: self.values[text] for text in texts}

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--suite", choices=("providers", "pipeline"), required=True)
    parser.add_argument(
        "--source-dir", type=Path, default=Path.home() / "OneDrive" / "Desktop"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("test") / "benchmarks")
    parser.add_argument("--images", nargs="+", type=int)
    parser.add_argument("--ocr", nargs="+", choices=("windows", "paddle"))
    parser.add_argument(
        "--providers", nargs="+", choices=tuple(PROVIDER_LABELS)
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any translation or pipeline case fails",
    )
    parser.add_argument(
        "--cached", action="store_true", help="Also measure exact-frame reuse"
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        help="Use saved Google translations to isolate CPU work",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    images = args.images or ([5] if args.suite == "providers" else [5, 6])
    engines = args.ocr or (
        ["windows"] if args.suite == "providers" else ["windows", "paddle"]
    )
    providers = args.providers or (
        list(PROVIDER_LABELS)
        if args.suite == "providers"
        else ["google"]
    )
    output = args.output_dir / args.phase
    output.mkdir(parents=True, exist_ok=True)

    import tkinter as tk
    from main import _enable_windows_dpi_awareness

    _enable_windows_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    overlay = ScreenOverlay(root)
    failures = []
    try:
        for engine_name in engines:
            setup_started = perf_counter()
            if engine_name == "windows":
                from modules.ocr.windows_ocr import WindowsOCRModule

                engine = WindowsOCRModule()
            else:
                from modules.ocr.paddle_ocr import PaddleOCRModule

                engine = PaddleOCRModule()
            ocr_setup_ms = (perf_counter() - setup_started) * 1000
            queue = Queue()
            worker = OCRWorker(
                engine,
                lambda lines, elapsed, token: queue.put(
                    ("result", lines, elapsed, token)
                ),
                lambda error, token: queue.put(("error", error, 0, token)),
            )
            try:
                for provider in providers:
                    for number in images:
                        path = next(args.source_dir.glob(f"исходное-{number}.*"))
                        load_started = perf_counter()
                        image = np.array(Image.open(path).convert("RGB"))
                        load_ms = (perf_counter() - load_started) * 1000
                        for repeat in range(args.repeat):
                            name = f"{engine_name}-{provider}-{number}-{repeat + 1}"
                            print(f"START {args.phase}/{name}", flush=True)
                            translator = (
                                ReplayTranslator(
                                    args.replay_dir
                                    / f"{engine_name}-{provider}-{number}-1.json"
                                )
                                if args.replay_dir
                                else Translator(provider=provider, hybrid_enabled=False)
                            )
                            runtime = RuntimeBundle(
                                (engine_name, provider, False, "argos", False),
                                worker,
                                queue,
                                translator,
                            )
                            cache = FrameCache() if args.cached else None
                            stats = []
                            send = httpx.Client.send

                            def instrument(client, request, *positional, **kwargs):
                                started = perf_counter()
                                try:
                                    response = send(
                                        client, request, *positional, **kwargs
                                    )
                                except Exception as error:
                                    stats.append(
                                        {
                                            "host": request.url.host,
                                            "error": type(error).__name__,
                                            "ms": round(
                                                (perf_counter() - started) * 1000, 2
                                            ),
                                        }
                                    )
                                    raise
                                stats.append(
                                    {
                                        "host": request.url.host,
                                        "status": response.status_code,
                                        "http": response.http_version,
                                        "retry_after": response.headers.get(
                                            "Retry-After"
                                        ),
                                        "ms": round(
                                            (perf_counter() - started) * 1000, 2
                                        ),
                                    }
                                )
                                return response

                            try:
                                bounds = (0, 0, image.shape[1], image.shape[0])
                                started = perf_counter()
                                with patch.object(httpx.Client, "send", instrument):
                                    result = process_capture(
                                        ImageCapture(image),
                                        bounds,
                                        False,
                                        runtime,
                                        cache,
                                    )
                                pipeline_ms = (perf_counter() - started) * 1000
                                preview = Image.fromarray(image).convert("RGBA")
                                photos = {}
                                photo = ImageTk.PhotoImage
                                create_image = overlay.canvas.create_image
                                rectangle = overlay.canvas.create_rectangle

                                def photo_spy(value=None, **kwargs):
                                    value = (
                                        value
                                        if value is not None
                                        else kwargs.pop("image")
                                    )
                                    item = photo(value, **kwargs)
                                    photos[id(item)] = value
                                    return item

                                def image_spy(x, y, **kwargs):
                                    value = photos[id(kwargs["image"])].convert("RGBA")
                                    preview.alpha_composite(value, (int(x), int(y)))
                                    return create_image(x, y, **kwargs)

                                def rectangle_spy(*box, **kwargs):
                                    x1, y1, x2, y2 = map(int, box)
                                    preview.paste(kwargs["fill"], (x1, y1, x2, y2))
                                    return rectangle(*box, **kwargs)

                                with ExitStack() as stack:
                                    stack.enter_context(
                                        patch.object(ImageTk, "PhotoImage", photo_spy)
                                    )
                                    stack.enter_context(
                                        patch.object(
                                            overlay.canvas, "create_image", image_spy
                                        )
                                    )
                                    stack.enter_context(
                                        patch.object(
                                            overlay.canvas,
                                            "create_rectangle",
                                            rectangle_spy,
                                        )
                                    )
                                    if args.profile:
                                        import cProfile

                                        profiler = cProfile.Profile()
                                        profiler.enable()
                                    shown = perf_counter()
                                    kwargs = {
                                        "backgrounds": getattr(
                                            result, "backgrounds", None
                                        )
                                    }
                                    if getattr(result, "overlay", None) is not None:
                                        kwargs["prepared"] = result.overlay
                                    display = overlay.show(
                                        result.translated_regions,
                                        bounds,
                                        result.source_image,
                                        **kwargs,
                                    )
                                    overlay_ms = (perf_counter() - shown) * 1000
                                    if args.profile:
                                        profiler.disable()
                                        profiler.dump_stats(
                                            str(output / f"{name}.prof")
                                        )
                                overlay.clear()
                                preview.convert("RGB").save(output / f"{name}.png")
                                record = {
                                    "phase": args.phase,
                                    "suite": args.suite,
                                    "ocr": engine_name,
                                    "provider": provider,
                                    "image": path.name,
                                    "source_sha256": hashlib.sha256(
                                        path.read_bytes()
                                    ).hexdigest(),
                                    "size": [image.shape[1], image.shape[0]],
                                    "repeat": repeat + 1,
                                    "ocr_setup_ms": round(ocr_setup_ms, 2),
                                    "image_load_ms": round(load_ms, 2),
                                    "pipeline_ms": round(pipeline_ms, 2),
                                    "ocr_ms": round(result.ocr_ms, 2),
                                    "translate_ms": round(result.translate_ms, 2),
                                    "processing_ms": round(
                                        getattr(
                                            result,
                                            "processing_ms",
                                            pipeline_ms
                                            - result.ocr_ms
                                            - result.translate_ms
                                            - result.background_ms
                                            - result.capture_ms,
                                        ),
                                        2,
                                    ),
                                    "background_ms": round(result.background_ms, 2),
                                    "overlay_prepare_ms": round(
                                        getattr(result, "overlay_prepare_ms", 0), 2
                                    ),
                                    "ui_ms": round(overlay_ms, 2),
                                    "total_ms": round(pipeline_ms + overlay_ms, 2),
                                    "recognized_lines": result.recognized_count,
                                    "filtered_lines": result.filtered_count,
                                    "translated_regions": len(
                                        result.translated_regions
                                    ),
                                    "display": asdict(display)
                                    if display is not None
                                    else None,
                                    "errors": result.errors,
                                    "translation_stats": asdict(translator.last_stats)
                                    if hasattr(translator, "last_stats")
                                    else None,
                                    "http_requests": stats,
                                    "http_statuses": dict(
                                        Counter(
                                            str(item.get("status", item.get("error")))
                                            for item in stats
                                        )
                                    ),
                                    "translations": [
                                        {"region": asdict(region), "text": text}
                                        for region, text in result.translated_regions
                                    ],
                                    "replayed_translation": args.replay_dir is not None,
                                }
                                if result.errors:
                                    failures.append(name)
                                if args.cached:
                                    started = perf_counter()
                                    cached = process_capture(
                                        ImageCapture(image),
                                        bounds,
                                        False,
                                        runtime,
                                        cache,
                                    )
                                    cached_pipeline_ms = (
                                        perf_counter() - started
                                    ) * 1000
                                    kwargs = {
                                        "backgrounds": getattr(
                                            cached, "backgrounds", None
                                        )
                                    }
                                    if getattr(cached, "overlay", None) is not None:
                                        kwargs["prepared"] = cached.overlay
                                    started = perf_counter()
                                    overlay.show(
                                        cached.translated_regions,
                                        bounds,
                                        cached.source_image,
                                        **kwargs,
                                    )
                                    cached_ui_ms = (perf_counter() - started) * 1000
                                    overlay.clear()
                                    record["cached"] = {
                                        "hit": cached.cache_hit,
                                        "pipeline_ms": round(cached_pipeline_ms, 2),
                                        "ui_ms": round(cached_ui_ms, 2),
                                    }
                                (output / f"{name}.json").write_text(
                                    json.dumps(record, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                                summary = {
                                    key: value
                                    for key, value in record.items()
                                    if key not in {"translations", "http_requests"}
                                }
                                print(
                                    json.dumps(summary, ensure_ascii=False), flush=True
                                )
                            except Exception as error:
                                failures.append(name)
                                (output / f"{name}-error.json").write_text(
                                    json.dumps(
                                        {
                                            "error": f"{type(error).__name__}: {error}",
                                            "http_requests": stats,
                                        },
                                        ensure_ascii=False,
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )
                                print(
                                    f"ERROR {name}: {type(error).__name__}: {error}",
                                    flush=True,
                                )
                            finally:
                                if hasattr(runtime, "translation_executor"):
                                    runtime.translation_executor.shutdown(
                                        wait=True, cancel_futures=True
                                    )
                                translator.close()
            finally:
                worker.close()
    finally:
        overlay.close()
        root.destroy()
    if args.strict and failures:
        print("Failed cases: " + ", ".join(failures), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
