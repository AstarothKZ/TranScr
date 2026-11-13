"""Windows click-through overlay with source-aligned text and repaired backgrounds."""

import ctypes
import logging
import os
import tkinter as tk

from PIL import ImageTk
from modules.overlay.preparation import DisplayStats, PreparedOverlay, prepare_overlay

LOGGER = logging.getLogger("transcr.pipeline")


class ScreenOverlay:
    """Draw translated text inside the original OCR rectangle without expanding it."""

    TRANSPARENT_COLOR = "#ff00ff"
    TEXT_COLOR = "#111111"
    BOX_COLOR = "#ffffff"

    def __init__(self, root: tk.Tk) -> None:
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.title("TranScr Overlay")
        self.window.overrideredirect(True)
        self.window.configure(background=self.TRANSPARENT_COLOR)
        self.window.attributes("-topmost", True)
        self.canvas = tk.Canvas(
            self.window,
            background=self.TRANSPARENT_COLOR,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
        )
        self.canvas.pack(fill="both", expand=True)
        self._images = []
        self._closed = False
        self.visible = False
        self.capture_excluded = False
        self._prepared = None
        self._screen_bounds = None
        self._native_ready = False

    def clear(self) -> None:
        """Hide immediately; retain just the last raster for cheap exact-frame reuse."""
        if not self._closed:
            self.window.withdraw()
            self.visible = False

    def show(self, translated_regions, screen_bounds, source_image=None, backgrounds=None,
             *, prepared: PreparedOverlay | None = None) -> DisplayStats:
        self.clear()
        left, top, width, height = screen_bounds
        if prepared is None:
            prepared = prepare_overlay(translated_regions, (width, height), source_image,
                                       backgrounds=backgrounds)
        if prepared.image is None:
            return prepared.stats
        if self._screen_bounds != screen_bounds:
            self.window.geometry(f"{width}x{height}+0+0")
            self.window.update_idletasks()
            if not self._native_ready:
                if not self._enable_click_through():
                    raise RuntimeError("Не удалось настроить окно оверлея Windows")
                self._native_ready = True
                self.capture_excluded = self._exclude_from_capture()
            if not self._set_native_window_bounds(left, top, width, height):
                raise RuntimeError("Не удалось задать положение оверлея Windows")
            self._screen_bounds = screen_bounds
        if self._prepared is not prepared:
            self.canvas.delete('all')
            photo = ImageTk.PhotoImage(prepared.image, master=self.window)
            self._images = [photo]
            self.canvas.create_image(prepared.left, prepared.top, anchor='nw', image=photo)
            self._prepared = prepared
        self.window.deiconify()
        self.visible = True
        self.window.update_idletasks()
        LOGGER.info("Оверлей показал %d блоков, %d строк; пропущено %d: %s",
                    prepared.stats.regions, prepared.stats.rows, prepared.stats.skipped,
                    ', '.join(prepared.skipped_reasons) or 'нет')
        return prepared.stats

    def _exclude_from_capture(self) -> bool:
        if os.name != 'nt':
            return False
        try:
            function = ctypes.windll.user32.SetWindowDisplayAffinity
            function.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            function.restype = ctypes.c_bool
            return bool(function(ctypes.c_void_p(self._native_hwnd()), 0x11))
        except (AttributeError, OSError):
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.visible = False
        self._images = []
        self._prepared = None
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def _native_hwnd(self) -> int:
        """Return the outer Win32 HWND, not Tk's inner client HWND."""
        frame_id = str(self.window.wm_frame())
        try:
            return int(frame_id, 0)
        except ValueError as error:
            raise RuntimeError(
                f"Не удалось получить HWND окна оверлея: {frame_id!r}"
            ) from error

    def _set_native_window_bounds(
        self, left: int, top: int, width: int, height: int
    ) -> bool:
        if os.name != "nt":
            return False
        try:
            user32 = getattr(ctypes, "windll").user32
            hwnd = self._native_hwnd()
            set_window_pos = getattr(user32, "SetWindowPos")
            set_window_pos.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            set_window_pos.restype = ctypes.c_bool
            flags = (
                0x0010 | 0x0020 | 0x0200
            )  # NOACTIVATE | FRAMECHANGED | NOOWNERZORDER
            success = set_window_pos(
                ctypes.c_void_p(hwnd),
                ctypes.c_void_p(-1),  # HWND_TOPMOST
                int(left),
                int(top),
                int(width),
                int(height),
                flags,
            )
            if not success:
                raise ctypes.WinError()
            return True
        except Exception:
            return False

    def _enable_click_through(self) -> bool:
        """Make the magenta window background transparent and pass mouse input through."""
        if os.name != "nt":
            return False
        try:
            user32 = getattr(ctypes, "windll").user32
            hwnd = self._native_hwnd()
            get_style = getattr(user32, "GetWindowLongPtrW", None) or getattr(
                user32, "GetWindowLongW"
            )
            set_style = getattr(user32, "SetWindowLongPtrW", None) or getattr(
                user32, "SetWindowLongW"
            )
            get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_style.restype = ctypes.c_ssize_t
            set_style.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
            set_style.restype = ctypes.c_ssize_t

            hwnd_arg = ctypes.c_void_p(hwnd)
            exstyle = get_style(hwnd_arg, -20)  # GWL_EXSTYLE
            exstyle |= 0x00080000  # WS_EX_LAYERED
            exstyle |= 0x00000020  # WS_EX_TRANSPARENT (mouse click-through)
            exstyle |= 0x00000080  # WS_EX_TOOLWINDOW
            exstyle |= 0x08000000  # WS_EX_NOACTIVATE
            set_style(hwnd_arg, -20, exstyle)

            set_layered_attributes = getattr(user32, "SetLayeredWindowAttributes")
            set_layered_attributes.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_ubyte,
                ctypes.c_uint,
            ]
            set_layered_attributes.restype = ctypes.c_bool
            success = set_layered_attributes(
                hwnd_arg,
                0x00FF00FF,
                0,
                0x00000001,  # LWA_COLORKEY
            )
            if not success:
                raise ctypes.WinError()
            return True
        except Exception:
            return False
