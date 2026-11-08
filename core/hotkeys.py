"""Match hotkeys against the event stream, without keyboard.is_pressed races."""

import ctypes
import os


def modifier_is_down(name):
    codes = {"alt": 0x12, "ctrl": 0x11, "control": 0x11, "shift": 0x10,
             "windows": 0x5B, "left alt": 0xA4, "right alt": 0xA5,
             "left ctrl": 0xA2, "right ctrl": 0xA3,
             "left shift": 0xA0, "right shift": 0xA1}
    if os.name != "nt" or name not in codes:
        return True
    return bool(ctypes.windll.user32.GetAsyncKeyState(codes[name]) & 0x8000)


class HotkeyRouter:
    def __init__(self, bindings, parse_hotkey, dismiss=None, modifier_state=None):
        self.modifier_state = modifier_state
        self.enabled = True
        self.bindings = []
        self.pressed = set()
        self.dismiss = dismiss
        for text, callback in bindings:
            steps = parse_hotkey(text)
            if len(steps) != 1:
                raise ValueError(
                    "Используйте одно сочетание клавиш без последовательностей через запятую"
                )
            self.bindings.append(([set(codes) for codes in steps[0]], callback, text.split("+")[-1].strip(), text.lower().split("+")[:-1]))

    def __call__(self, event):
        if not self.enabled:
            return
        code = event.scan_code
        if event.event_type == "up":
            self.pressed.discard(code)
            return
        if event.event_type != "down" or code in self.pressed:
            return
        self.pressed.add(code)
        for keys, callback, trigger, modifiers in self.bindings:
            if self.modifier_state is not None and not all(self.modifier_state(m.strip()) for m in modifiers):
                continue
            # Windows scan codes can name several logical keys (e.g. numpad
            # navigation). A digit binding must actually be a digit event.
            if trigger.isdecimal() and getattr(event, "name", None) != trigger:
                continue
            # Trigger on the final key, never when pressing a modifier after it.
            if code not in keys[-1]:
                continue
            if all(self.pressed & alternatives for alternatives in keys) and any(
                code in alternatives for alternatives in keys
            ):
                callback()
                return
        if self.dismiss is not None:
            self.dismiss(event)
