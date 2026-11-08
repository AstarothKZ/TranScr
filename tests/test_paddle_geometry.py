"""Paddle adapter checks without loading model binaries in the unit suite."""

import importlib.util
import json
import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


class PaddleGeometryTests(unittest.TestCase):
    def test_tight_detection_preserves_rows_and_separates_heading(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads(
            (root / "tests/fixtures/paddle_magazine.json").read_text(encoding="utf-8")
        )
        package = Mock()
        package.PaddleOCR.return_value.predict.return_value = [Mock(json={"res": data})]
        spec = importlib.util.spec_from_file_location(
            "paddle_adapter_test", root / "modules/ocr/paddle_ocr.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"paddleocr": package}):
            spec.loader.exec_module(module)
        adapter = module.PaddleOCRModule()
        self.assertEqual(
            package.PaddleOCR.call_args.kwargs["text_det_unclip_ratio"], 1.0
        )
        lines = adapter.recognize(np.zeros((1100, 800, 3), dtype=np.uint8))
        self.assertEqual(len(lines), 27)
        heading = next(line for line in lines if line.text.startswith("is very"))
        subtitle = next(line for line in lines if line.text.startswith("That's why"))
        self.assertLess(heading.bottom, subtitle.top)
        self.assertEqual(heading.bottom, 577)
        from modules.text_processing.text_regions import group_ocr_texts
        regions = group_ocr_texts(lines)
        left_body = [r for r in regions if r.left < 100 and 600 < r.top < 800]
        self.assertEqual([len(r.lines) for r in left_body], [4, 6])
        right_body = next(r for r in regions if r.text.startswith("The RGS 2 may"))
        self.assertEqual(len(right_body.lines), 4)
        dimensions = next(r for r in regions if r.text.startswith("Sight Head"))
        self.assertEqual(len(dimensions.lines), 1)
