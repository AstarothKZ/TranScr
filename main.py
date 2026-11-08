import ctypes
import os

def _enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return
    try:
        getattr(windll.shcore, "SetProcessDpiAwareness")(2)  # per-monitor DPI awareness
    except Exception:
        try:
            getattr(windll.user32, "SetProcessDPIAware")()
        except Exception:
            pass

def main() -> None:
    _enable_windows_dpi_awareness()
    # Import UI modules only after configuring Windows pixel coordinates.
    from core.application import Application

    Application().run()

if __name__ == "__main__":
    main()
