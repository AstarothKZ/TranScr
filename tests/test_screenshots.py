"""PNG exports and application actions, without OCR/translation network calls."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import Mock, patch

import keyboard
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.config import AppConfig
from core.hotkeys import HotkeyRouter
from utilities.screenshots import compose_screenshot, save_screenshot, screenshot_prefix


class ScreenshotTests(unittest.TestCase):
    def test_names_cover_all_providers_and_inactive_state(self):
        self.assertEqual(screenshot_prefix(), "empty")
        for ocr, label in (("windows", "win"), ("paddle", "paddle")):
            for provider, name in (
                ("google", "goog"),
                ("yandex", "yand"),
                ("argos", "argos"),
            ):
                self.assertEqual(
                    screenshot_prefix((ocr, provider, False, "argos", False)),
                    f"{label}-{name}",
                )

    def test_export_composes_offsets_and_alpha_without_mutating_source(self):
        source = np.full((8, 10, 3), (0, 0, 200), dtype=np.uint8)
        original = source.copy()
        overlay = SimpleNamespace(
            image=Image.new("RGBA", (3, 2), (200, 0, 0, 128)), left=4, top=3
        )
        result = compose_screenshot(source, overlay)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (10, 8))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 200))
        self.assertEqual(result.getpixel((4, 3)), (100, 0, 100))
        np.testing.assert_array_equal(source, original)

    def test_plain_image_and_empty_overlay_preserve_source(self):
        source = np.full((2, 3, 3), (20, 40, 60), dtype=np.uint8)
        for overlay in (None, SimpleNamespace(image=None)):
            np.testing.assert_array_equal(
                np.array(compose_screenshot(source, overlay)), source
            )

    def test_series_survive_restart_and_do_not_overwrite_files(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            image = Image.new("RGB", (4, 3), "red")
            old = folder / "paddle-goog-3.png"
            old.write_bytes(b"existing")
            self.assertEqual(
                save_screenshot(image, folder, "paddle-goog").name, "paddle-goog-4.png"
            )
            self.assertEqual(
                save_screenshot(image, folder, "paddle-goog").name, "paddle-goog-5.png"
            )
            self.assertEqual(
                save_screenshot(image, folder, "win-yand").name, "win-yand-1.png"
            )
            self.assertEqual(
                save_screenshot(image, folder, "empty").name, "empty-1.png"
            )
            self.assertEqual(old.read_bytes(), b"existing")
            with Image.open(folder / "paddle-goog-4.png") as saved:
                self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0))

    def test_concurrent_saves_reserve_different_names(self):
        with TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool:
            folder = Path(directory)
            futures = [
                pool.submit(save_screenshot, Image.new("RGB", (2, 2)), folder, "empty")
                for _ in range(8)
            ]
            names = {future.result().name for future in futures}
            self.assertEqual(names, {f"empty-{n}.png" for n in range(1, 9)})

    def test_failed_save_removes_only_its_reserved_file(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "empty-1.png").write_bytes(b"existing")
            image = Mock()
            image.save.side_effect = OSError("disk full")
            with self.assertRaises(OSError):
                save_screenshot(image, folder, "empty")
            self.assertEqual([path.name for path in folder.iterdir()], ["empty-1.png"])

    def test_prefix_cannot_escape_output_directory(self):
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            save_screenshot(Image.new("RGB", (2, 2)), Path(directory), "../outside")

    def test_optional_hotkey_config_is_persisted_and_old_options_are_ignored(self):
        with TemporaryDirectory() as directory:
            config = AppConfig(Path(directory))
            config.path = Path(directory) / "settings.json"
            self.assertEqual(config.load()["screenshot_hotkey"], "")
            config.save({"screenshot_hotkey": "F8"})
            self.assertEqual(config.load()["screenshot_hotkey"], "f8")
            config.save({"screenshot_hotkey": ""})
            self.assertEqual(config.load()["screenshot_hotkey"], "")
            config.path.write_text(
                json.dumps({"screenshot_hotkey": None, "unused_old_option": True}),
                encoding="utf-8",
            )
            self.assertEqual(config.load()["screenshot_hotkey"], "")
            self.assertNotIn("unused_old_option", config.load())

    def test_quick_area_hotkey_can_be_empty_after_reload(self):
        with TemporaryDirectory() as directory:
            config = AppConfig(Path(directory))
            config.path = Path(directory) / "settings.json"
            for value in ("", "  ", None):
                config.save({"quick_area_hotkey": value})
                self.assertEqual(config.load()["quick_area_hotkey"], "")

    def test_hotkey_match_does_not_call_any_key_dismiss(self):
        save, dismiss = Mock(), Mock()
        router = HotkeyRouter([("f8", save)], keyboard.parse_hotkey, dismiss)
        code = keyboard.key_to_scan_codes("f8")[0]
        router(SimpleNamespace(scan_code=code, event_type="down", name="f8"))
        router(SimpleNamespace(scan_code=code, event_type="down", name="f8"))
        save.assert_called_once()
        dismiss.assert_not_called()

    def test_duplicate_physical_hotkeys_are_rejected(self):
        with self.assertRaises(ValueError):
            HotkeyRouter(
                [("ctrl+s", Mock()), ("control+s", Mock())], keyboard.parse_hotkey
            )

    def test_capture_pipeline_and_frame_cache_work_without_automatic_saving(self):
        from core.pipeline import FrameCache, process_capture
        from modules.ocr.types import OCRText

        image = Image.new("RGB", (360, 90), "white")
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        ImageDraw.Draw(image).text(
            (12, 12), "Save screenshot sample", font=font, fill="black"
        )
        pixels = np.array(image)
        queue = Queue()
        worker = Mock()

        def submit(frame, token):
            queue.put(
                (
                    "result",
                    [OCRText("Save screenshot sample", 10, 10, 300, 42)],
                    1,
                    token,
                )
            )
            return True

        worker.submit.side_effect = submit
        capture = Mock()
        capture.capture_region.return_value = pixels
        translator = Mock(last_stats=None)
        translator.translate_many.side_effect = lambda texts, cancel: {
            text: "Сохранить снимок" for text in texts
        }
        with ThreadPoolExecutor(1) as executor:
            runtime = SimpleNamespace(
                ocr_worker=worker,
                ocr_results=queue,
                translator=translator,
                translation_executor=executor,
            )
            cache = FrameCache()
            result = process_capture(capture, (0, 0, 360, 90), runtime, cache)
            cached = process_capture(capture, (0, 0, 360, 90), runtime, cache)
        self.assertEqual(result.errors, [])
        self.assertIsNotNone(result.overlay.image)
        self.assertTrue(cached.cache_hit)
        self.assertIs(cached.source_image, result.source_image)
        worker.submit.assert_called_once()
        translator.translate_many.assert_called_once()


class ApplicationScreenshotTests(unittest.TestCase):
    def setUp(self):
        from core.application import Application

        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        config = AppConfig(self.folder)
        config.path = self.folder / "settings.json"
        with patch("core.application.AppConfig", return_value=config):
            self.app = Application()
        self.app.root.update()
        self.app.root.withdraw()
        self.app.screenshot_dir = self.folder / "screenshots"
        self.addCleanup(self.close)

    def close(self):
        self.app.root.update_idletasks()
        self.app.shutdown()
        self.app._screenshot_executor.shutdown(wait=True)
        self.app._capture_executor.shutdown(wait=True)

    def wait_for_save(self):
        deadline = perf_counter() + 4
        while self.app._saving_screenshot and perf_counter() < deadline:
            self.app.root.update()
            sleep(0.01)
        self.assertFalse(
            self.app._saving_screenshot, "Screenshot action did not finish"
        )

    def test_visible_overlay_exports_actual_raster_without_screen_recapture(self):
        from core.pipeline import CaptureResult
        from modules.overlay.preparation import DisplayStats, PreparedOverlay

        app = self.app
        pixels = np.full((40, 80, 3), (0, 0, 200), dtype=np.uint8)
        prepared = PreparedOverlay(
            Image.new("RGBA", (20, 10), (255, 0, 0, 255)), 5, 6, DisplayStats(1, 1, 0)
        )
        app._active = True
        app._runtime_settings = ("paddle", "google", False, "argos", False)
        app._last_capture_result = CaptureResult(
            (-100, 10, 80, 40),
            [],
            1,
            0,
            1,
            1,
            1,
            [],
            source_image=pixels,
            overlay=prepared,
        )
        app.overlay.show([], app._last_capture_result.area, pixels, prepared=prepared)
        with patch("core.application.ScreenCapturer") as capture:
            app._save_screenshot()
            self.wait_for_save()
            capture.assert_not_called()
        with Image.open(app.screenshot_dir / "paddle-goog-1.png") as result:
            self.assertEqual(result.size, (80, 40))
            self.assertEqual(result.getpixel((5, 6)), (255, 0, 0))
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 200))
        self.assertTrue(app.overlay.visible)
        self.assertTrue(app.overlay.capture_excluded)

    def test_stopped_or_hidden_overlay_uses_fresh_frame_with_correct_prefix(self):
        app = self.app
        app._runtime_settings = ("windows", "yandex", False, "argos", False)
        for active, name in ((False, "empty-1.png"), (True, "win-yand-1.png")):
            app._active = active
            app.overlay.clear()
            with patch("core.application.ScreenCapturer") as capture:
                capture.return_value.capture_region.return_value = np.full(
                    (3, 4, 3), (0, 255, 0), dtype=np.uint8
                )
                app._save_screenshot()
                self.wait_for_save()
                capture.return_value.capture_region.assert_called_once_with(
                    app.selected_area
                )
                capture.return_value.close.assert_called_once()
            with Image.open(app.screenshot_dir / name) as result:
                self.assertEqual(result.getpixel((0, 0)), (0, 255, 0))

    def test_failure_is_reported_and_save_button_recovers(self):
        app = self.app
        with (
            patch("core.application.ScreenCapturer") as capture,
            patch("core.application.save_screenshot", side_effect=OSError("disk full")),
        ):
            capture.return_value.capture_region.return_value = np.zeros(
                (3, 4, 3), dtype=np.uint8
            )
            app._save_screenshot()
            self.wait_for_save()
        self.assertIn("disk full", app.status_var.get())
        self.assertEqual(str(app.save_screenshot_button.cget("state")), "normal")

    def test_save_hotkey_works_when_stopped_and_survives_stop(self):
        app = self.app
        app.screenshot_hotkey_var.set("f8")
        with (
            patch("core.application.keyboard.hook"),
            patch("core.application.keyboard.unhook"),
        ):
            app._save_settings()
            self.assertEqual(len(app._hotkey_router.bindings), 1)
            app._runtime_settings = ("windows", "google", False, "argos", False)
            app._activate_hotkeys()
            self.assertTrue(app._active)
            self.assertEqual(len(app._hotkey_router.bindings), 3)
            app.stop()
            self.assertFalse(app._active)
            self.assertEqual(len(app._hotkey_router.bindings), 1)
            code = keyboard.key_to_scan_codes("f8")[0]
            with patch("core.application.ScreenCapturer") as capture:
                capture.return_value.capture_region.return_value = np.zeros(
                    (2, 2, 3), dtype=np.uint8
                )
                app._hotkey_router(
                    SimpleNamespace(scan_code=code, event_type="down", name="f8")
                )
                deadline = perf_counter() + 4
                while (
                    not (app.screenshot_dir / "empty-1.png").exists()
                    and perf_counter() < deadline
                ):
                    app.root.update()
                    sleep(0.01)
                self.wait_for_save()
            self.assertTrue((app.screenshot_dir / "empty-1.png").exists())
            app.screenshot_hotkey_var.set("")
            app._save_settings()
            self.assertIsNone(app._keyboard_hook)

    def test_conflicting_optional_hotkey_has_clear_error(self):
        app = self.app
        app.screenshot_hotkey_var.set("alt+t")
        with patch("core.application.keyboard.hook") as hook:
            app._configure_idle_hotkeys()
            hook.assert_not_called()
        self.assertIn("должно отличаться", app.status_var.get())

    def test_translation_starts_without_quick_area_or_screenshot_hotkeys(self):
        app = self.app
        app.quick_area_hotkey_var.set("")
        app.screenshot_hotkey_var.set("")
        app._runtime_settings = ("windows", "google", False, "argos", False)
        with (
            patch("core.application.keyboard.hook"),
            patch("core.application.keyboard.unhook"),
        ):
            app._activate_hotkeys()
            self.assertTrue(app._active)
            self.assertEqual(len(app._hotkey_router.bindings), 1)
            self.assertNotIn("выбрать и перевести", app.status_var.get())
            app.stop()

    def test_area_controls_remain_visible_at_minimum_window_size(self):
        app = self.app
        app.root.deiconify()
        app.root.geometry("480x620")
        app.root.update()
        for name in (
            "select_area_button",
            "full_screen_button",
            "save_screenshot_button",
            "start_button",
            "stop_button",
        ):
            widget = getattr(app.ui, name)
            self.assertTrue(widget.winfo_ismapped(), name)
            while widget is not app.root:
                parent = widget.master
                self.assertGreaterEqual(
                    widget.winfo_rootx(), parent.winfo_rootx(), name
                )
                self.assertGreaterEqual(
                    widget.winfo_rooty(), parent.winfo_rooty(), name
                )
                self.assertLessEqual(
                    widget.winfo_rootx() + widget.winfo_width(),
                    parent.winfo_rootx() + parent.winfo_width(),
                    name,
                )
                self.assertLessEqual(
                    widget.winfo_rooty() + widget.winfo_height(),
                    parent.winfo_rooty() + parent.winfo_height(),
                    name,
                )
                widget = parent


if __name__ == "__main__":
    unittest.main()
