"""Windows click-through overlay with source-aligned text and repaired backgrounds."""

import ctypes
import logging
import os
import tkinter as tk
from dataclasses import replace

from PIL import ImageTk
from modules.overlay.fonts import PixelFont
from modules.overlay.styling import preserve_case, estimate_bold, strengthen_color
from modules.overlay.background import prepare_background
from modules.overlay.rendering import render_row

from modules.text_processing.text_regions import layout_slots, separate_row_overlaps

LOGGER = logging.getLogger("transcr.pipeline")


class ScreenOverlay:
    """Draw translated text inside the original OCR rectangle without expanding it."""

    TRANSPARENT_COLOR = "#ff00ff"
    TEXT_COLOR = "#111111"
    BOX_COLOR = "#ffffff"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
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

    def clear(self) -> None:
        """Hide the overlay and remove all prior translated blocks."""
        if self._closed:
            return
        self.window.withdraw()
        self.canvas.delete("all")
        self._images = []

    def show(self, translated_regions, screen_bounds, source_image=None, backgrounds=None) -> int:
        """Draw source-bound translations and return the number of shown regions."""
        from modules.overlay.layout import fit_region

        self.clear()
        left, top, width, height = screen_bounds
        if width <= 0 or height <= 0:
            raise ValueError(f"Некорректные размеры экрана: {screen_bounds}")
        self.window.geometry(f"{width}x{height}+0+0")
        self.window.update_idletasks()
        occupied = []
        accepted = []
        accepted_regions = 0
        prepared_layers = {}
        prepared_styles = {}
        fonts = {}

        def make_font(size):
            key = (size, bold)
            if key not in fonts:
                fonts[key] = PixelFont(size, bold=bold)
            return fonts[key]
        # Apply across region boundaries as well, including cached OCR output.
        sources = [
            line for region, _ in translated_regions for line in layout_slots(region)
        ]
        corrected = iter(separate_row_overlaps(sources))
        translated_regions = [
            (
                replace(
                    region, lines=tuple(next(corrected) for _ in layout_slots(region))
                ),
                text,
            )
            for region, text in translated_regions
        ]
        # Deterministic overlap policy; a whole region is accepted or rejected.
        regions = sorted(
            translated_regions,
            key=lambda item: (
                -sum(line.confidence or 0 for line in item[0].lines)
                / max(1, len(item[0].lines)),
                item[0].top,
                item[0].left,
            ),
        )
        for region, text in regions:
            if not text.strip():
                continue
            text = preserve_case(region.text, text)
            bold = estimate_bold(source_image, region)
            plan = fit_region(region, text, make_font)
            boxes = [
                (row.source.left, row.source.top, row.source.right, row.source.bottom)
                for row in plan
            ]
            if any(
                x1 < region.left
                or x2 > region.right
                or y1 < region.top
                or y2 > region.bottom
                for x1, y1, x2, y2 in boxes
            ):
                LOGGER.warning(
                    "Оверлей отклонил строки за границами своей области (%s, %s)",
                    region.left,
                    region.top,
                )
                continue
            if not plan or any(
                x1 < 0 or y1 < 0 or x2 > width or y2 > height
                for x1, y1, x2, y2 in boxes
            ):
                LOGGER.warning(
                    "Оверлей пропустил область (%s, %s): перевод не помещается",
                    region.left,
                    region.top,
                )
                continue
            if any(
                self._overlap_ratio(box, used) > 0 for box in boxes for used in occupied
            ):
                LOGGER.warning(
                    "Оверлей пропустил пересекающуюся область (%s, %s)",
                    region.left,
                    region.top,
                )
                continue
            # Prepare and validate the whole block before changing any source
            # pixels. Rendering errors must never leave a background-only box.
            region_layers = {}
            region_styles = {}
            try:
                for row, box in zip(plan, boxes):
                    style = backgrounds.get(box) if backgrounds is not None else None
                    if style is None and source_image is not None:
                        style = prepare_background(source_image, box)
                    color = strengthen_color(style.text_color, style.color) if style else self.TEXT_COLOR
                    region_layers[id(row)] = render_row(row, color)
                    region_styles[id(row)] = style
            except (ValueError, OSError) as error:
                LOGGER.warning('Оверлей сохранил исходный блок (%s, %s): %s',region.left,region.top,error)
                continue
            prepared_layers.update(region_layers)
            prepared_styles.update(region_styles)
            accepted.extend(plan)
            accepted_regions += 1
            occupied.extend(boxes)
        for row in accepted:
            line = row.source
            box = (line.left, line.top, line.right, line.bottom)
            style = prepared_styles[id(row)]
            if style is not None and style.patch is not None:
                photo = ImageTk.PhotoImage(style.patch, master=self.window)
                self._images.append(photo)
                self.canvas.create_image(line.left, line.top, anchor="nw", image=photo)
            else:
                color = style.color if style else self.BOX_COLOR
                self.canvas.create_rectangle(*box, fill=color, outline=color, width=0)
        for row in accepted:
            photo = ImageTk.PhotoImage(prepared_layers[id(row)], master=self.window)
            self._images.append(photo)
            self.canvas.create_image(row.source.left, row.source.top, anchor='nw', image=photo)
        if not accepted:
            return 0
        if not self._enable_click_through() or not self._set_native_window_bounds(
            left, top, width, height
        ):
            raise RuntimeError("Не удалось настроить окно оверлея Windows")
        self.window.deiconify()
        self.window.attributes("-topmost", True)
        self.window.update_idletasks()
        LOGGER.info("Оверлей показал %d строк в исходных рамках", len(accepted))
        return accepted_regions

    @staticmethod
    def _overlap_ratio(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        """Intersection area as a fraction of the smaller source-line area."""
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        if not intersection:
            return 0.0
        area_first = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
        area_second = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
        return intersection / min(area_first, area_second)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._images = []
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
