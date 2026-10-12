"""A simple Windows click-through overlay for translated OCR regions."""

import ctypes
import os
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Sequence

from .text_blocks import TextRegion


class ScreenOverlay:
    """Draw opaque white boxes with translated text over their source regions."""

    TRANSPARENT_COLOR = "#ff00ff"
    TEXT_COLOR = "#111111"
    BOX_COLOR = "#ffffff"
    PADDING = 4

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(background=self.TRANSPARENT_COLOR)
        self.window.attributes("-topmost", True)
        try:
            # On Windows, Tk uses this color as the transparent window key.
            self.window.attributes("-transparentcolor", self.TRANSPARENT_COLOR)
        except tk.TclError:
            # The application targets Windows 10+, but retain a safe fallback
            # for development platforms where Tk has no transparent-color key.
            self.window.attributes("-alpha", 0.96)

        self.canvas = tk.Canvas(
            self.window,
            background=self.TRANSPARENT_COLOR,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
        )
        self.canvas.pack(fill="both", expand=True)
        self.window.bind("<Escape>", lambda _event: self.clear())
        self._fonts: list[tkfont.Font] = []
        self._closed = False

    def hide(self) -> None:
        """Hide the overlay before a new screenshot is taken."""
        if not self._closed:
            self.window.withdraw()

    def clear(self) -> None:
        """Hide the overlay and remove the previous translated blocks."""
        if self._closed:
            return
        self.window.withdraw()
        self.canvas.delete("all")
        self._fonts.clear()

    def show(
        self,
        translated_regions: Sequence[tuple[TextRegion, str]],
        screen_bounds: tuple[int, int, int, int],
    ) -> None:
        """Display translated regions over the screenshot's virtual desktop.

        `screen_bounds` is (desktop_left, desktop_top, width, height), matching
        the MSS virtual-screen capture including multi-monitor offsets.
        """
        self.clear()
        if not translated_regions:
            return

        desktop_left, desktop_top, width, height = screen_bounds
        if width <= 0 or height <= 0:
            raise ValueError(f"Некорректные размеры экрана: {screen_bounds}")

        self.window.geometry(f"{width}x{height}+0+0")
        self.window.update_idletasks()

        text_items: list[tuple[TextRegion, str, int]] = []
        fonts: list[tkfont.Font] = []
        for region, translated_text in translated_regions:
            text = translated_text.strip()
            if not text:
                continue

            pad = self.PADDING
            region_width = max(24, region.right - region.left)
            text_width = max(24, region_width - pad * 2)
            font = tkfont.Font(
                root=self.window,
                family="Segoe UI",
                # Negative Tk font size means pixels (OCR geometry is in pixels).
                size=-region.font_size,
            )
            fonts.append(font)
            text_id = self.canvas.create_text(
                max(0, region.left) + pad,
                max(0, region.top) + pad,
                anchor="nw",
                text=text,
                fill=self.TEXT_COLOR,
                font=font,
                width=text_width,
                justify="left",
            )
            text_items.append((region, text, text_id))

        self.window.update_idletasks()
        for region, _text, text_id in text_items:
            text_bounds = self.canvas.bbox(text_id)
            if text_bounds is None:
                continue

            pad = self.PADDING
            x1 = max(0, min(region.left, text_bounds[0] - pad))
            y1 = max(0, min(region.top, text_bounds[1] - pad))
            x2 = min(width, max(region.right, text_bounds[2] + pad))
            y2 = min(height, max(region.bottom, text_bounds[3] + pad))
            background_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=self.BOX_COLOR,
                outline=self.BOX_COLOR,
                width=1,
            )
            # Put each background behind all translated text, not over it.
            self.canvas.tag_lower(background_id)

        # tkinter.Font deletes its native font when the Python object is freed.
        self._fonts = fonts
        self.window.deiconify()
        self.window.attributes("-topmost", True)
        self.window.update_idletasks()
        positioned = self._set_native_window_bounds(
            desktop_left, desktop_top, width, height
        )
        click_through = self._enable_click_through()
        if not positioned or not click_through:
            # Fail closed: never leave a visible overlay that may block clicks.
            self.window.withdraw()
            raise RuntimeError(
                "Не удалось настроить прозрачное click-through окно Windows; "
                "оверлей скрыт, чтобы не блокировать управление приложениями."
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def _set_native_window_bounds(
        self, left: int, top: int, width: int, height: int
    ) -> bool:
        """Use Win32 coordinates so monitors left/above the primary work too."""
        if os.name != "nt":
            return False
        try:
            user32 = ctypes.windll.user32
            user32.SetWindowPos.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.SetWindowPos.restype = ctypes.c_bool
            hwnd = self.window.winfo_id()
            flags = 0x0010 | 0x0040 | 0x0020  # NOACTIVATE | SHOWWINDOW | FRAMECHANGED
            success = user32.SetWindowPos(
                ctypes.c_void_p(hwnd),
                ctypes.c_void_p(-1),  # HWND_TOPMOST
                int(left), int(top), int(width), int(height), flags,
            )
            if not success:
                raise ctypes.WinError()
            return True
        except Exception as error:
            print(f"Предупреждение: не удалось точно расположить оверлей: {error}", flush=True)
            return False

    def _enable_click_through(self) -> bool:
        """Mark the native window as layered, transparent to clicks and no-activate."""
        if os.name != "nt":
            return False
        try:
            user32 = ctypes.windll.user32
            hwnd = self.window.winfo_id()
            get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_style.restype = ctypes.c_ssize_t
            set_style.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
            set_style.restype = ctypes.c_ssize_t

            exstyle = get_style(ctypes.c_void_p(hwnd), -20)  # GWL_EXSTYLE
            exstyle |= 0x00080000  # WS_EX_LAYERED
            exstyle |= 0x00000020  # WS_EX_TRANSPARENT
            exstyle |= 0x00000080  # WS_EX_TOOLWINDOW
            exstyle |= 0x08000000  # WS_EX_NOACTIVATE
            set_style(ctypes.c_void_p(hwnd), -20, exstyle)
            user32.SetWindowPos.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.SetWindowPos.restype = ctypes.c_bool
            success = user32.SetWindowPos(
                ctypes.c_void_p(hwnd), ctypes.c_void_p(-1), 0, 0, 0, 0,
                0x0001 | 0x0002 | 0x0010 | 0x0020,
            )  # NOSIZE | NOMOVE | NOACTIVATE | FRAMECHANGED
            if not success:
                raise ctypes.WinError()
            return True
        except Exception as error:
            print(f"Предупреждение: не удалось включить режим click-through: {error}", flush=True)
            return False
