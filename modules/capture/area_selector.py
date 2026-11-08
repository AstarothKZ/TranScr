"""Mouse drag selector for choosing a rectangle across the virtual desktop."""

import ctypes
import os
import tkinter as tk
from collections.abc import Callable
from ctypes import wintypes

from .types import ScreenBounds


class AreaSelector:
    """A temporary screen-sized window that returns an absolute screen rectangle."""

    def __init__(
        self,
        root: tk.Tk,
        screen_bounds: ScreenBounds,
        on_selected: Callable[[ScreenBounds], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.root = root
        self.screen_left, self.screen_top, self.screen_width, self.screen_height = (
            screen_bounds
        )
        self.on_selected = on_selected
        self.on_cancel = on_cancel
        self._start: tuple[int, int] | None = None
        self._rect_id = None
        self._closed = False

        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(background="#202020")
        self.window.attributes("-topmost", True)
        # This is a deliberate dimming layer used only while the user is
        # selecting an area; unlike the translation overlay it receives clicks.
        self.window.attributes("-alpha", 0.28)
        self.window.geometry(f"{self.screen_width}x{self.screen_height}+0+0")

        self.canvas = tk.Canvas(
            self.window,
            background="#202020",
            highlightthickness=0,
            borderwidth=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            self.screen_width // 2,
            max(32, self.screen_height // 20),
            text="Зажмите левую кнопку мыши и выделите область для перевода · Esc — отмена",
            fill="#ffffff",
            font=("Segoe UI", 14, "bold"),
            anchor="n",
        )
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish_drag)
        self.window.bind("<Escape>", self.cancel)
        self.window.bind("<Button-3>", self.cancel)

        self.window.update_idletasks()
        self._position_on_virtual_desktop()
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.canvas.focus_set()

    def _position_on_virtual_desktop(self) -> None:
        if os.name != "nt":
            return
        try:
            user32 = getattr(ctypes, "windll").user32
            hwnd = int(str(self.window.wm_frame()), 0)
            set_window_pos = getattr(user32, "SetWindowPos")
            set_window_pos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            set_window_pos.restype = wintypes.BOOL
            success = set_window_pos(
                wintypes.HWND(hwnd),
                wintypes.HWND(-1),  # HWND_TOPMOST
                self.screen_left,
                self.screen_top,
                self.screen_width,
                self.screen_height,
                0x0020 | 0x0200,  # FRAMECHANGED | NOOWNERZORDER
            )
            if not success:
                raise ctypes.WinError()
        except Exception:
            # Tk's geometry fallback still works on common single-monitor setups.
            pass

    def _start_drag(self, event) -> None:
        self._start = (event.x, event.y)
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#2fffd0",
            width=3,
            fill="#2f8fff",
            stipple="gray25",
        )

    def _drag(self, event) -> None:
        if self._start is None or self._rect_id is None:
            return
        self.canvas.coords(
            self._rect_id, self._start[0], self._start[1], event.x, event.y
        )

    def _finish_drag(self, event) -> None:
        if self._start is None:
            return
        x1, y1 = self._start
        x2, y2 = event.x, event.y
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        self._start = None
        if width < 20 or height < 12:
            self.cancel()
            return

        selected: ScreenBounds = (
            self.screen_left + left,
            self.screen_top + top,
            width,
            height,
        )
        self._destroy()
        self.on_selected(selected)

    def cancel(self, _event=None) -> None:
        if self._closed:
            return
        self._destroy()
        if self.on_cancel is not None:
            self.on_cancel()

    def _destroy(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass
