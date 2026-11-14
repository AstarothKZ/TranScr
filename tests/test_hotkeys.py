import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import keyboard

from core.application import Application
from core.hotkeys import HotkeyRouter, modifier_is_down


class HotkeyTests(unittest.TestCase):
    def screenshot_application(self, hotkey):
        app = Application.__new__(Application)
        for name, value in (
            ("screenshot_hotkey", hotkey),
            ("start_hotkey", "alt+t"),
            ("quick_area_hotkey", "alt+q"),
            ("dismiss_mode", "any"),
        ):
            setattr(app, name + "_var", SimpleNamespace(get=lambda value=value: value))
        return app

    def test_sided_modifiers_remain_parseable_and_trigger(self):
        for modifier in ("left ctrl", "right alt", "left shift", "right windows"):
            with self.subTest(modifier=modifier):
                app = self.screenshot_application(f"  {modifier.upper()} + s  ")
                hotkey = app._screenshot_hotkey()
                callback = Mock()
                router = HotkeyRouter([(hotkey, callback)], keyboard.parse_hotkey)
                for alternatives in keyboard.parse_hotkey(hotkey)[0]:
                    router(SimpleNamespace(event_type="down", scan_code=alternatives[0], name="s"))
                callback.assert_called_once_with()

    def test_screenshot_conflict_ignores_spacing_and_case(self):
        app = self.screenshot_application("  ALT + T  ")
        with self.assertRaises(ValueError):
            app._screenshot_hotkey()

    def test_generic_windows_modifier_accepts_right_windows(self):
        key_state = Mock(side_effect=lambda code: 0x8000 if code == 0x5C else 0)
        windll = SimpleNamespace(user32=SimpleNamespace(GetAsyncKeyState=key_state))
        with (
            patch("core.hotkeys.os.name", "nt"),
            patch("core.hotkeys.ctypes.windll", windll, create=True),
        ):
            self.assertTrue(modifier_is_down("windows"))
            self.assertTrue(modifier_is_down("right windows"))
            self.assertFalse(modifier_is_down("left windows"))

    def test_key_repeat_does_not_repeat_command(self):
        callback = Mock()
        router = HotkeyRouter([("alt+t", callback)], keyboard.parse_hotkey)
        alt, trigger = keyboard.parse_hotkey("alt+t")[0]
        router(SimpleNamespace(event_type="down", scan_code=alt[0], name="alt"))
        event = SimpleNamespace(event_type="down", scan_code=trigger[0], name="t")
        router(event)
        router(event)
        callback.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
