"""Isolate provider cold, uncached warm, and cached latency on one OCR payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

sys.path.insert(
    0,
    os.environ.get("TRANSCR_BENCH_CODE_ROOT", str(Path(__file__).resolve().parents[1])),
)

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from modules.ocr.windows_ocr import WindowsOCRModule
from modules.text_processing.filters import should_translate
from modules.text_processing.geometry import prepare_lines
from modules.text_processing.image_validation import has_source_ink
from modules.text_processing.text_regions import group_ocr_texts
from modules.translation.catalog import PROVIDER_LABELS
from modules.translation.translator import Translator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path.home() / "OneDrive" / "Desktop" / "исходное-6.png",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=tuple(PROVIDER_LABELS),
        default=list(PROVIDER_LABELS),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("test/benchmarks"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    frame = np.array(Image.open(args.image).convert("RGB"))
    engine = WindowsOCRModule()
    started = perf_counter()
    recognized = engine.recognize(frame)
    ocr_ms = (perf_counter() - started) * 1000
    valid = prepare_lines(
        frame, [line for line in recognized if has_source_ink(frame, line)]
    )
    meaningful = [line for line in valid if should_translate(line.text)]
    regions = group_ocr_texts(meaningful, context=valid)
    texts = list(
        dict.fromkeys(
            region.text.strip() for region in regions if should_translate(region.text)
        )
    )
    record = {
        "phase": args.phase,
        "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "ocr_ms": round(ocr_ms, 2),
        "ocr_lines": len(recognized),
        "texts": texts,
        "payload_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode()
        ).hexdigest(),
        "runs": [],
    }
    failures = []
    for provider in args.providers:
        translator = Translator(provider=provider)
        try:
            for mode in ("cold", "warm_uncached", "cached"):
                if mode == "warm_uncached":
                    if hasattr(translator, "clear_cache"):
                        translator.clear_cache()
                    else:
                        # An explicit baseline code root can predate the public cache reset API.
                        with translator._cache_lock:
                            translator._cache.clear()
                            translator._cache_chars = 0
                started = perf_counter()
                values = translator.translate_many(texts)
                elapsed = (perf_counter() - started) * 1000
                errors = {
                    text: f"{type(value).__name__}: {value}"
                    for text, value in values.items()
                    if isinstance(value, Exception)
                }
                run = {
                    "provider": provider,
                    "mode": mode,
                    "elapsed_ms": round(elapsed, 2),
                    "stats": asdict(translator.last_stats),
                    "errors": errors,
                    "translations": {
                        text: value
                        for text, value in values.items()
                        if isinstance(value, str)
                    },
                }
                record["runs"].append(run)
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in run.items()
                            if key not in {"translations", "errors"}
                        }
                        | {"errors": len(errors)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if errors:
                    failures.append((provider, mode))
        finally:
            translator.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.phase}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
