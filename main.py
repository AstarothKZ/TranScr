import ctypes
import os

# Reduce noisy logs from native OCR libraries where supported.
os.environ.setdefault("GLOG_minloglevel", "2")


def _enable_windows_dpi_awareness() -> None:
    """Keep MSS pixel coordinates aligned with the overlay on scaled displays."""
    if os.name != "nt":
        return
    try:
        # PER_MONITOR_DPI_AWARE: use physical pixel coordinates on each monitor.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_enable_windows_dpi_awareness()

from core.application import Application


if __name__ == "__main__":
    Application().run()
