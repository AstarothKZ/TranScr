import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from core.hotkeys import HotkeyRouter


class HotkeyTests(unittest.TestCase):
    def test_three_bindings_repeat_release_and_dismiss(self):
        capture, quick, reset, dismiss = Mock(), Mock(), Mock(), Mock()
        codes = {
            "alt+t": (((56, 100), (20,)),),
            "alt+q": (((56, 100), (16,)),),
            "esc": (((1,),),),
        }
        router = HotkeyRouter(
            [("alt+t", capture), ("alt+q", quick), ("esc", reset)],
            codes.__getitem__,
            dismiss,
        )

        def event(code, kind="down"):
            router(SimpleNamespace(scan_code=code, event_type=kind, name="key"))

        event(56)
        event(20)
        event(20)  # auto-repeat must not enqueue another capture
        event(20, "up")
        event(16)
        event(16, "up")
        event(56, "up")
        event(1)
        capture.assert_called_once()
        quick.assert_called_once()
        reset.assert_called_once()
        event(1, "up")
        event(45)
        self.assertEqual(dismiss.call_args.args[0].scan_code, 45)

    def test_right_modifier_and_event_order(self):
        callback = Mock()
        router = HotkeyRouter([("alt+t", callback)], lambda _: (((56, 100), (20,)),))
        for code, kind in [
            (100, "down"),
            (20, "down"),
            (100, "up"),
            (20, "up"),
            (100, "down"),
            (20, "down"),
        ]:
            router(SimpleNamespace(scan_code=code, event_type=kind))
        self.assertEqual(callback.call_count, 2)

    def test_reject_sequences(self):
        with self.assertRaises(ValueError):
            HotkeyRouter([("sequence", Mock())], lambda _: (((1,),), ((2,),)))

    def test_digit_scan_alias_does_not_trigger_during_typing(self):
        callback = Mock()
        router = HotkeyRouter([('1', callback)], lambda _: (((2, 79),),))
        for name in ('end', 'a', '!'):
            router(SimpleNamespace(scan_code=79, event_type='down', name=name))
            router(SimpleNamespace(scan_code=79, event_type='up', name=name))
        callback.assert_not_called()
        router(SimpleNamespace(scan_code=2, event_type='down', name='1'))
        callback.assert_called_once()

    def test_modifier_pressed_after_trigger_does_not_capture(self):
        callback = Mock()
        router = HotkeyRouter([('alt+t', callback)], lambda _: (((56,), (20,)),))
        router(SimpleNamespace(scan_code=20, event_type='down'))
        router(SimpleNamespace(scan_code=56, event_type='down'))
        callback.assert_not_called()

    def test_lost_alt_release_does_not_capture(self):
        callback = Mock()
        state = Mock(return_value=True)
        router = HotkeyRouter([('alt+t', callback)], lambda _: (((56,), (20,)),), modifier_state=state)
        router(SimpleNamespace(scan_code=56, event_type='down'))
        state.return_value = False
        router(SimpleNamespace(scan_code=20, event_type='down'))
        callback.assert_not_called()

    def test_disabled_old_router_ignores_pending_events(self):
        callback = Mock()
        router = HotkeyRouter([('t', callback)], lambda _: (((20,),),))
        router.enabled = False
        router(SimpleNamespace(scan_code=20, event_type='down'))
        callback.assert_not_called()
