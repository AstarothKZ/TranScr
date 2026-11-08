"""Tkinter settings window and background screen-translation pipeline."""

import logging
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from time import perf_counter
from tkinter import messagebox

import keyboard

from core.config import AppConfig
from core.hotkeys import HotkeyRouter, modifier_is_down
from core.pipeline import CaptureResult, FrameCache, RuntimeBundle, process_capture
from core.session_logging import SessionFileHandler
from modules.capture.area_selector import AreaSelector
from modules.capture.screen_capture import ScreenCapturer
from modules.capture.types import ScreenBounds
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText
from modules.overlay.screen_overlay import ScreenOverlay
from modules.translation.catalog import PROVIDER_LABELS
from modules.translation.translator import Translator
from utilities.ui.settings_window import SettingsWindow


class QueueLogHandler(logging.Handler):
    """Forward log records to a queue; the Tk widget is updated on the UI thread."""

    def __init__(self, target_queue: Queue) -> None:
        super().__init__(logging.INFO)
        self.target_queue = target_queue
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.target_queue.put((self.format(record), record.levelno))
        except Exception:
            self.handleError(record)


class Application:
    """Coordinates the UI, global hotkeys and an asynchronous capture pipeline."""

    HOTKEY = "alt+t"
    DEFAULT_OCR_LANGUAGE = "en-US"
    PROVIDER_LABELS = PROVIDER_LABELS
    PROVIDER_VALUES = {label: key for key, label in PROVIDER_LABELS.items()}
    OCR_LABELS = {
        "Windows OCR": "windows",
        "PaddleOCR (дополнительная установка)": "paddle",
    }
    OCR_VALUES = {value: label for label, value in OCR_LABELS.items()}
    MODIFIER_KEYS = {
        "alt",
        "left alt",
        "right alt",
        "ctrl",
        "left ctrl",
        "right ctrl",
        "shift",
        "left shift",
        "right shift",
        "windows",
        "left windows",
        "right windows",
    }

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        screenshot_dir = self.project_root / "test" / "screenshots"

        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.screen_capturer = ScreenCapturer(screenshot_dir)
        self.full_screen_bounds = self.screen_capturer.virtual_bounds
        self.config = AppConfig(self.project_root)
        self._saved_settings = self.config.load()
        saved_area = self._saved_settings.get("area")
        self.selected_area = self._restore_area(saved_area, self.full_screen_bounds)
        self.area_is_custom = (
            saved_area is not None and self.selected_area != self.full_screen_bounds
        )
        self.overlay = ScreenOverlay(self.root)

        self._events: Queue = Queue()
        self._capture_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transcr-pipeline"
        )
        self._frame_cache = FrameCache()
        self._runtime: RuntimeBundle | None = None
        self._runtime_settings: tuple[str, str, bool, str] | None = None
        self._starting = False
        self._active = False
        self._busy = False
        self._shutting_down = False
        self._display_generation = 0
        self._capture_id = 0
        self._capture_started = 0.0
        self._runtime_publish_lock = threading.Lock()
        self._capture_cancel = threading.Event()
        self._keyboard_hook = None
        self._quick_area_hotkey_text = "alt+q"
        self._capture_hotkey_text = self.HOTKEY
        self._area_selector: AreaSelector | None = None

        self._ui_log_queue: Queue = Queue()
        self._ui_log_handler: QueueLogHandler | None = None
        self._pipeline_log_handler: QueueLogHandler | None = None
        self._pipeline_logger: logging.Logger | None = None
        self._dependency_log_handler: QueueLogHandler | None = None
        self._setup_logger()
        self._build_ui()
        self._append_log("Приложение запущено. Настройки загружены.")
        self.root.after(20, self._poll_events)

    def _setup_logger(self) -> None:
        """Log to a new session file and forward messages to the in-app journal."""
        self._logger = logging.getLogger("transcr.application")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # Make Application instances independent (useful for tests and restarting).
        for handler in list(self._logger.handlers):
            if isinstance(handler, (QueueLogHandler, SessionFileHandler)):
                self._logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        self._session_file_handler = None
        try:
            self._session_file_handler = SessionFileHandler(self.project_root / "logs")
            self._logger.addHandler(self._session_file_handler)
        except OSError:
            self._logger.exception("Не удалось создать файл журнала")

        self._ui_log_handler = QueueLogHandler(self._ui_log_queue)
        self._logger.addHandler(self._ui_log_handler)

        self._pipeline_logger = logging.getLogger("transcr.pipeline")
        self._pipeline_logger.setLevel(logging.INFO)
        self._pipeline_logger.propagate = False
        for handler in list(self._pipeline_logger.handlers):
            if isinstance(handler, (QueueLogHandler, SessionFileHandler)):
                self._pipeline_logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        self._pipeline_log_handler = QueueLogHandler(self._ui_log_queue)
        self._pipeline_logger.addHandler(self._pipeline_log_handler)
        for handler in self._logger.handlers:
            if isinstance(handler, SessionFileHandler):
                self._pipeline_logger.addHandler(handler)

        # Forward dependency warnings/errors (for example Argos or OCR warnings).
        # App messages do not duplicate here because the app logger doesn't propagate.
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if isinstance(handler, (QueueLogHandler, SessionFileHandler)):
                root_logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        self._dependency_log_handler = QueueLogHandler(self._ui_log_queue)
        self._dependency_log_handler.setLevel(logging.WARNING)
        root_logger.addHandler(self._dependency_log_handler)
        if self._session_file_handler is not None:
            root_logger.addHandler(self._session_file_handler)

    def _build_ui(self) -> None:
        initial_area_text = (
            "Область: " + self._area_description(self.selected_area)
            if self.area_is_custom
            else "Весь виртуальный экран: " + self._area_description(self.selected_area)
        )
        self.ui = SettingsWindow(
            self.root,
            saved_settings=self._saved_settings,
            ocr_labels=self.OCR_LABELS,
            ocr_values=self.OCR_VALUES,
            provider_labels=self.PROVIDER_LABELS,
            provider_values=self.PROVIDER_VALUES,
            initial_area_text=initial_area_text,
            callbacks={
                "select_area": self.select_area,
                "select_full_screen": self.select_full_screen,
                "start": self.start,
                "capture": lambda: self._events.put(("capture_request", "button")),
                "stop": self.stop,
                "save_settings": self._save_settings,
            },
        )
        # Explicit aliases keep controller references visible to IDE/type checkers.
        self.ocr_var = self.ui.ocr_var
        self.provider_var = self.ui.provider_var
        self.fallback_provider_var = self.ui.fallback_provider_var
        self.hybrid_var = self.ui.hybrid_var
        self.start_hotkey_var = self.ui.start_hotkey_var
        self.quick_area_hotkey_var = self.ui.quick_area_hotkey_var
        self.dismiss_mode_var = self.ui.dismiss_mode_var
        self.dismiss_hotkey_var = self.ui.dismiss_hotkey_var
        self.area_var = self.ui.area_var
        self.save_debug_var = self.ui.save_debug_var
        self.status_var = self.ui.status_var
        self.ocr_combo = self.ui.ocr_combo
        self.provider_combo = self.ui.provider_combo
        self.fallback_combo = self.ui.fallback_combo
        self.hybrid_button = self.ui.hybrid_button
        self.select_area_button = self.ui.select_area_button
        self.start_button = self.ui.start_button
        self.capture_button = self.ui.capture_button
        self.stop_button = self.ui.stop_button

    def _append_log(self, text: str, level: int | None = None) -> None:
        """Record an application event; a queue handler mirrors it into the UI log tab."""
        logger = getattr(self, "_logger", None)
        if logger is None:
            return
        if level is None:
            lowered = text.casefold()
            if (
                "ошибка" in lowered
                or "не удалось" in lowered
                or lowered.startswith("сбой")
            ):
                level = logging.ERROR
            elif "предупреждение" in lowered or "пропущено" in lowered:
                level = logging.WARNING
            else:
                level = logging.INFO
        logger.log(level, "%s", text)

    @staticmethod
    def _restore_area(saved_area, screen_bounds: ScreenBounds) -> ScreenBounds:
        if not saved_area:
            return screen_bounds
        sx, sy, sw, sh = screen_bounds
        left, top, width, height = map(int, saved_area)
        right = min(sx + sw, left + width)
        bottom = min(sy + sh, top + height)
        left = max(sx, left)
        top = max(sy, top)
        if right - left < 20 or bottom - top < 12:
            return screen_bounds
        return left, top, right - left, bottom - top

    @staticmethod
    def _area_description(bounds: ScreenBounds) -> str:
        left, top, width, height = bounds
        return f"X={left}, Y={top} · {width} × {height} px"

    def _save_settings(self) -> None:
        try:
            ocr_key = self.OCR_LABELS.get(self.ocr_var.get(), "windows")
            provider_key = self.PROVIDER_VALUES.get(self.provider_var.get(), "mymemory")
            fallback_key = self.PROVIDER_VALUES.get(
                self.fallback_provider_var.get(), "argos"
            )
            if fallback_key == provider_key:
                fallback_key = "argos" if provider_key != "argos" else "mymemory"
            area_value = list(self.selected_area) if self.area_is_custom else None
            self.config.save(
                {
                    "ocr": ocr_key,
                    "provider": provider_key,
                    "hybrid_enabled": bool(self.hybrid_var.get()),
                    "fallback_provider": fallback_key,
                    "start_hotkey": self.start_hotkey_var.get().strip().lower(),
                    "quick_area_hotkey": self.quick_area_hotkey_var.get()
                    .strip()
                    .lower(),
                    "dismiss_mode": self.dismiss_mode_var.get(),
                    "dismiss_hotkey": self.dismiss_hotkey_var.get().strip().lower(),
                    "area": area_value,
                    "save_debug_screenshots": bool(self.save_debug_var.get()),
                }
            )
        except Exception as error:
            self._append_log(f"Не удалось сохранить настройки: {error}")

    def select_area(self) -> None:
        if self._busy or self._starting or self._area_selector is not None:
            self.status_var.set(
                "Дождитесь завершения текущей операции перед выбором области."
            )
            return
        self.overlay.clear()
        if self._active:
            self._display_generation += 1
        self.status_var.set("Выделите прямоугольник мышью. Esc — отмена.")
        self._append_log("Начат выбор области захвата мышью.")
        self.select_area_button.configure(state="disabled")

        def selected(bounds: ScreenBounds) -> None:
            self._area_selector = None
            self.selected_area = bounds
            self.area_is_custom = True
            self.area_var.set("Область: " + self._area_description(bounds))
            self.select_area_button.configure(state="normal")
            self._save_settings()
            self._append_log(f"Область выбрана: {self._area_description(bounds)}")
            self.status_var.set(
                f"Область сохранена. Нажмите «Запустить переводчик» или {self.start_hotkey_var.get().strip()}."
            )

        def cancelled() -> None:
            self._area_selector = None
            self.select_area_button.configure(state="normal")
            self.status_var.set("Выбор области отменён.")
            self._append_log("Выбор области захвата отменён.")

        try:
            self._area_selector = AreaSelector(
                self.root,
                self.full_screen_bounds,
                on_selected=selected,
                on_cancel=cancelled,
            )
        except Exception as error:
            self.select_area_button.configure(state="normal")
            self.status_var.set(f"Не удалось открыть выбор области: {error}")
            self._append_log(f"Ошибка выбора области: {error}")

    def quick_select_area(self) -> None:
        """Select a one-shot capture without touching persistent area settings."""
        if (
            not self._active
            or self._busy
            or self._starting
            or self._area_selector is not None
        ):
            return
        self.overlay.clear()
        self._display_generation += 1
        self.root.iconify()
        self.status_var.set("Выделите область для разового перевода. Esc — отмена.")

        def selected(bounds: ScreenBounds) -> None:
            self._area_selector = None
            self._start_capture("quick-area", area_override=bounds)

        def cancelled() -> None:
            self._area_selector = None
            self.status_var.set(
                "Разовый перевод отменён. Сохранённая область не изменена."
            )

        try:
            self._area_selector = AreaSelector(
                self.root,
                self.full_screen_bounds,
                on_selected=selected,
                on_cancel=cancelled,
            )
        except Exception as error:
            self._area_selector = None
            self._append_log(f"Ошибка быстрого выбора области: {error}")

    def select_full_screen(self) -> None:
        if self._busy or self._starting:
            self.status_var.set("Дождитесь завершения текущей операции.")
            return
        self.overlay.clear()
        if self._active:
            self._display_generation += 1
        self.selected_area = self.full_screen_bounds
        self.area_is_custom = False
        self.area_var.set(
            "Весь виртуальный экран: " + self._area_description(self.selected_area)
        )
        self._save_settings()
        self._append_log("Выбрана область: весь экран.")
        self.status_var.set("Будет захватываться весь виртуальный экран.")

    def start(self) -> None:
        if self._shutting_down or self._starting:
            return
        if self._busy:
            self.status_var.set(
                "Идёт перевод. Дождитесь завершения перед повторным запуском."
            )
            return
        provider = self.PROVIDER_VALUES.get(self.provider_var.get(), "mymemory")
        fallback = self.PROVIDER_VALUES.get(self.fallback_provider_var.get(), "argos")
        self._append_log(
            f"Запуск: OCR={self.OCR_LABELS.get(self.ocr_var.get(), 'windows')}, "
            f"переводчик={provider}, гибрид={'включён' if self.hybrid_var.get() else 'выключен'}."
        )
        if fallback == provider:
            self.status_var.set("Основной и резервный переводчики должны отличаться.")
            return
        settings = (
            self.OCR_LABELS[self.ocr_var.get()],
            provider,
            bool(self.hybrid_var.get()),
            fallback,
        )
        start_hotkey = self.start_hotkey_var.get().strip()
        if not start_hotkey:
            self.status_var.set("Укажите сочетание клавиш для запуска перевода.")
            return
        if self.dismiss_mode_var.get() == "hotkey":
            dismiss_hotkey = self.dismiss_hotkey_var.get().strip()
            if not dismiss_hotkey:
                self.status_var.set("Укажите сочетание клавиш для сброса перевода.")
                return
            if dismiss_hotkey.lower().replace(" ", "") == start_hotkey.lower().replace(
                " ", ""
            ):
                self.status_var.set("Сочетания запуска и сброса должны отличаться.")
                return
        self._save_settings()
        if self._runtime is not None and self._runtime_settings == settings:
            self._activate_hotkeys()
            return

        self._starting = True
        self.ui.set_configuration_enabled(False)
        self.start_button.configure(state="disabled")
        self.select_area_button.configure(state="disabled")
        self.status_var.set(
            "Подготовка OCR и переводчика… При первой загрузке может потребоваться время."
        )
        self._append_log("Инициализация компонентов…")
        setup_thread = threading.Thread(
            target=self._prepare_runtime,
            args=(settings,),
            name="transcr-setup",
            daemon=True,
        )
        setup_thread.start()

    def _prepare_runtime(self, settings: tuple[str, str, bool, str]) -> None:
        ocr_kind, provider, hybrid_enabled, fallback_provider = settings
        new_ocr_worker = None
        translator = None
        try:
            if self._runtime is not None:
                try:
                    self._runtime.ocr_worker.close(timeout=1.5)
                except Exception:
                    pass
                try:
                    self._runtime.translator.close()
                except Exception:
                    pass
                self._runtime = None

            if ocr_kind == "windows":
                from modules.ocr.windows_ocr import WindowsOCRModule

                ocr_module = WindowsOCRModule(language=self.DEFAULT_OCR_LANGUAGE)
            elif ocr_kind == "paddle":
                from modules.ocr.paddle_ocr import PaddleOCRModule

                ocr_module = PaddleOCRModule()
            else:
                raise ValueError(f"Неизвестный OCR: {ocr_kind}")

            ocr_results: Queue = Queue(maxsize=2)

            def on_result(
                texts: list[OCRText], elapsed_ms: float, token: float
            ) -> None:
                try:
                    ocr_results.put_nowait(("result", texts, elapsed_ms, token))
                except Exception:
                    pass

            def on_error(ocr_error: Exception, token: float) -> None:
                try:
                    ocr_results.put_nowait(("error", ocr_error, 0.0, token))
                except Exception:
                    pass

            new_ocr_worker = OCRWorker(
                ocr_module, on_result=on_result, on_error=on_error
            )
            translator = Translator(
                root=self.root,
                from_code="en",
                to_code="ru",
                provider=provider,
                hybrid_enabled=hybrid_enabled,
                fallback_provider=fallback_provider,
            )
            argos_ready = True
            if provider == "argos":
                argos_ready = translator.check_language()

            runtime = RuntimeBundle(
                settings=settings,
                ocr_worker=new_ocr_worker,
                ocr_results=ocr_results,
                translator=translator,
                argos_model_ready=argos_ready,
            )
            with self._runtime_publish_lock:
                if self._shutting_down:
                    self._close_runtime(runtime)
                    return
                self._events.put(("runtime_ready", runtime))
        except Exception as error:
            self._logger.error(
                "Ошибка подготовки компонентов OCR/перевода", exc_info=True
            )
            if new_ocr_worker is not None:
                try:
                    new_ocr_worker.close(timeout=1.0)
                except Exception:
                    pass
            if translator is not None:
                try:
                    translator.close()
                except Exception:
                    pass
            self._events.put(("runtime_error", error))

    def _activate_hotkeys(self) -> None:
        runtime_settings = self._runtime_settings
        if runtime_settings is None:
            self.ui.set_configuration_enabled(True)
            self.start_button.configure(state="normal")
            self.status_var.set(
                "Компоненты ещё не готовы. Нажмите «Запустить переводчик» ещё раз."
            )
            return
        if self._active:
            self.status_var.set(
                f"Переводчик уже запущен. {self._capture_hotkey_text} — захват экрана."
            )
            return
        if self._busy:
            self.ui.set_configuration_enabled(True)
            self.status_var.set(
                "Сейчас обрабатывается захват; повторный запуск пока недоступен."
            )
            return
        self._remove_hotkeys()
        start_hotkey = self.start_hotkey_var.get().strip().lower().replace(" ", "")
        dismiss_mode = self.dismiss_mode_var.get()
        dismiss_hotkey = self.dismiss_hotkey_var.get().strip().lower().replace(" ", "")
        quick_hotkey = self.quick_area_hotkey_var.get().strip().lower().replace(" ", "")
        if (
            not quick_hotkey
            or quick_hotkey == start_hotkey
            or (dismiss_mode == "hotkey" and quick_hotkey == dismiss_hotkey)
        ):
            self.ui.set_configuration_enabled(True)
            self.start_button.configure(state="normal")
            self.status_var.set(
                "Сочетание быстрого выбора должно быть задано и отличаться от запуска и сброса."
            )
            return
        if not start_hotkey:
            self.ui.set_configuration_enabled(True)
            self.status_var.set("Сочетание запуска не задано.")
            self.start_button.configure(state="normal")
            return
        if dismiss_mode == "hotkey" and (
            not dismiss_hotkey or dismiss_hotkey == start_hotkey
        ):
            self.ui.set_configuration_enabled(True)
            self.status_var.set(
                "Сочетание сброса должно быть задано и отличаться от запуска."
            )
            self.start_button.configure(state="normal")
            return
        try:
            self._capture_hotkey_text = start_hotkey
            self._quick_area_hotkey_text = quick_hotkey
            bindings = [
                (start_hotkey, lambda: self._events.put(("capture_request", "hotkey"))),
                (quick_hotkey, lambda: self._events.put(("quick_area_request",))),
            ]
            if dismiss_mode == "hotkey":
                bindings.append(
                    (
                        dismiss_hotkey,
                        lambda: self._events.put(("dismiss", dismiss_hotkey, True)),
                    )
                )
            token = object()
            self._hotkey_session = token
            # Queue the action with its registration identity; stop/restart
            # invalidates commands still waiting on the UI thread.
            bindings = [(name, lambda action=action: self._events.put(("hotkey_command", token, action)))
                        for name, action in bindings]
            self._hotkey_router = HotkeyRouter(
                bindings,
                keyboard.parse_hotkey,
                self._dismiss_unbound_key if dismiss_mode == "any" else None,
                modifier_state=modifier_is_down,
            )
            self._keyboard_hook = keyboard.hook(self._hotkey_router, suppress=False)
        except Exception as error:
            self._remove_hotkeys()
            self.ui.set_configuration_enabled(True)
            self.status_var.set(f"Не удалось включить горячие клавиши: {error}")
            self._append_log(f"Ошибка горячих клавиш: {error}")
            self.start_button.configure(state="normal")
            return

        self._active = True
        self.ui.set_configuration_enabled(False)
        self.start_button.configure(text="Запущено", state="disabled")
        self.stop_button.configure(state="normal")
        self.capture_button.configure(state="normal")
        self.ocr_combo.configure(state="disabled")
        self.provider_combo.configure(state="disabled")
        self.fallback_combo.configure(state="disabled")
        self.hybrid_button.configure(state="disabled")
        self.select_area_button.configure(state="normal")
        reset_text = (
            "любая клавиша"
            if self.dismiss_mode_var.get() == "any"
            else self.dismiss_hotkey_var.get().strip()
        )
        self.status_var.set(
            f"Работает. {self._capture_hotkey_text} — перевод; {quick_hotkey} — выбрать и перевести; {reset_text} — сброс."
        )
        mode_text = self.PROVIDER_LABELS[runtime_settings[1]]
        if runtime_settings[2]:
            mode_text += f" → резерв: {self.PROVIDER_LABELS[runtime_settings[3]]}"
        self._append_log(f"Переводчик запущен: {mode_text}.")

    def _dismiss_unbound_key(self, event) -> None:
        if (
            self._active
            and self._area_selector is None
            and (event.name or "").lower() not in self.MODIFIER_KEYS
        ):
            self._events.put(("dismiss", event.name or "", True))

    def stop(self) -> None:
        if self._starting:
            self.status_var.set("Подождите, пока завершится подготовка компонентов.")
            return
        if not self._active:
            return
        self._active = False
        self._display_generation += 1
        self._capture_cancel.set()
        if self._area_selector is not None:
            self._area_selector.cancel()
        self._remove_hotkeys()
        self.overlay.clear()
        self.start_button.configure(text="Запустить переводчик", state="normal")
        self.stop_button.configure(state="disabled")
        self.capture_button.configure(state="disabled")
        self.ocr_combo.configure(state="readonly")
        self.ui.set_configuration_enabled(True)
        self.status_var.set(
            "Переводчик остановлен. Настройки можно изменить и запустить снова."
        )
        self._append_log("Переводчик остановлен.")
        try:
            self.root.deiconify()
            self.root.lift()
        except tk.TclError:
            pass

    def _remove_hotkeys(self) -> None:
        self._hotkey_session = None
        router = getattr(self, "_hotkey_router", None)
        if router is not None:
            router.enabled = False
            router.pressed.clear()
        if self._keyboard_hook is not None:
            try:
                keyboard.unhook(self._keyboard_hook)
            except Exception:
                pass
            self._keyboard_hook = None

    def _start_capture(
        self, source: str, area_override: ScreenBounds | None = None
    ) -> None:
        if self._area_selector is not None:
            return
        if not self._active:
            self.status_var.set("Сначала нажмите «Запустить переводчик».")
            return
        if self._busy:
            self.status_var.set(
                "Предыдущий захват ещё обрабатывается; повторный пропущен."
            )
            return
        if self._runtime is None:
            self.status_var.set("Компоненты ещё не готовы.")
            return

        self._busy = True
        self._capture_started = perf_counter()
        self._capture_cancel = threading.Event()
        self._capture_id += 1
        capture_id = self._capture_id
        self._append_log(
            f"Захват #{capture_id} запрошен ({source}); область {self._area_description(area_override or self.selected_area)}."
        )
        generation = self._display_generation
        area = area_override or self.selected_area
        save_debug = bool(self.save_debug_var.get())
        runtime = self._runtime
        self.overlay.clear()
        self.status_var.set("Захват экрана → OCR → перевод…")

        # Minimize our own settings window and allow DWM to repaint before MSS
        # captures the selected area. This prevents the settings UI being OCR'd.
        try:
            if self.root.state() == "normal":
                self.root.iconify()
        except tk.TclError:
            pass
        self.root.after(
            160,
            lambda: self._submit_capture(
                capture_id, generation, area, save_debug, runtime
            ),
        )

    def _submit_capture(
        self,
        capture_id: int,
        generation: int,
        area: ScreenBounds,
        save_debug: bool,
        runtime: RuntimeBundle,
    ) -> None:
        if (
            self._shutting_down
            or not self._active
            or generation != self._display_generation
        ):
            self._busy = False
            return
        future = self._capture_executor.submit(
            process_capture,
            self.screen_capturer,
            area,
            save_debug,
            runtime,
            self._frame_cache,
            self._capture_cancel,
        )

        def finished(done_future) -> None:
            try:
                result = done_future.result()
            except Exception as error:
                self._logger.error(
                    "Ошибка pipeline при захвате #%s", capture_id, exc_info=True
                )
                self._events.put(("capture_error", capture_id, generation, error))
            else:
                self._events.put(("capture_done", capture_id, generation, result))

        future.add_done_callback(finished)

    def _drain_ui_logs(self) -> None:
        """Drain log records on Tk's main thread and append them to the Journal tab."""
        log_queue = getattr(self, "_ui_log_queue", None)
        ui = getattr(self, "ui", None)
        if log_queue is None or ui is None:
            return
        while True:
            try:
                message, level = log_queue.get_nowait()
            except Empty:
                break
            try:
                ui.append_log(str(message), int(level))
            except tk.TclError:
                if self._shutting_down:
                    return
            except Exception:
                # A diagnostics view must never be able to stop the main event loop.
                continue

    def _poll_events(self) -> None:
        """Drain background events without ever stopping Tk's polling loop."""
        if self._shutting_down:
            return
        self._drain_ui_logs()
        try:
            while True:
                try:
                    item = self._events.get_nowait()
                except Empty:
                    break

                try:
                    kind = item[0]
                    if kind == "hotkey_command":
                        if self._active and item[1] is getattr(self, "_hotkey_session", None):
                            item[2]()
                    elif kind == "runtime_ready":
                        self._handle_runtime_ready(item[1])
                    elif kind == "runtime_error":
                        self._starting = False
                        self.ui.set_configuration_enabled(True)
                        self.start_button.configure(state="normal")
                        self.select_area_button.configure(state="normal")
                        self.status_var.set(
                            f"Не удалось подготовить приложение: {item[1]}"
                        )
                        self._append_log(f"Ошибка запуска: {item[1]}")
                    elif kind == "capture_request":
                        self._start_capture(item[1])
                    elif kind == "quick_area_request":
                        self.quick_select_area()
                    elif kind == "dismiss":
                        _kind, _key_name, cancel_pending = item
                        if cancel_pending:
                            self._display_generation += 1
                            self._capture_cancel.set()
                        self.overlay.clear()
                        self._append_log(f"Перевод сброшен клавишей: {_key_name}.")
                        if self._active:
                            self.status_var.set(
                                f"Перевод скрыт. {self._capture_hotkey_text} — новый захват."
                            )
                    elif kind == "capture_done":
                        self._handle_capture_done(item[1], item[2], item[3])
                    elif kind == "capture_error":
                        _kind, _capture_id, generation, error = item
                        self._busy = False
                        if self._active and generation == self._display_generation:
                            self.status_var.set(f"Ошибка обработки: {error}")
                        self._append_log(f"Ошибка захвата: {error}")
                except tk.TclError:
                    if self._shutting_down:
                        return
                    # A window may have been closed while a worker was finishing.
                    continue
                except Exception as error:
                    # Keep polling after a bad event so the Start button and status
                    # updates cannot silently stop working for the rest of the run.
                    try:
                        self.status_var.set(f"Ошибка обработки события: {error}")
                        self._append_log(f"Ошибка обработки события: {error}")
                    except tk.TclError:
                        if self._shutting_down:
                            return
        finally:
            if not self._shutting_down:
                try:
                    self.root.after(20, self._poll_events)
                except tk.TclError:
                    pass

    def _handle_runtime_ready(self, runtime: RuntimeBundle) -> None:
        self._starting = False
        self._frame_cache.clear()
        self._runtime = runtime
        self._runtime_settings = runtime.settings
        self.start_button.configure(state="normal")
        self.select_area_button.configure(state="normal")
        if not runtime.argos_model_ready:
            accepted = messagebox.askyesno(
                "Модель Argos не найдена",
                "Не найдена офлайн-модель перевода en → ru. Выбрать файл .argosmodel сейчас?",
                parent=self.root,
            )
            if accepted and runtime.translator.manual_model_installation():
                runtime.argos_model_ready = runtime.translator.check_language()
            if not runtime.argos_model_ready:
                self._close_runtime(runtime)
                self._runtime = None
                self._runtime_settings = None
                self.ui.set_configuration_enabled(True)
                self.start_button.configure(state="normal")
                self.status_var.set(
                    "Модель Argos не установлена. Выберите другой переводчик или установите модель."
                )
                self._append_log(
                    "Запуск не активирован: отсутствует модель Argos en → ru."
                )
                return
        self._activate_hotkeys()

    def _handle_capture_done(
        self, capture_id: int, generation: int, result: CaptureResult
    ) -> None:
        self._busy = False
        if result.screenshot_path is not None:
            self._append_log(
                f"Скриншот для отладки: {result.screenshot_path.as_posix()}"
            )
        if result.errors:
            for error in result.errors[:6]:
                self._append_log(f"Перевод пропущен: {error}")
            if len(result.errors) > 6:
                self._append_log(f"И ещё ошибок: {len(result.errors) - 6}")

        should_display = (
            self._active
            and generation == self._display_generation
            and not self._shutting_down
        )
        if not should_display:
            self._append_log(f"Задача #{capture_id} завершена, отображение отменено.")
            return

        if result.translated_regions:
            overlay_started = perf_counter()
            try:
                self.overlay.show(result.translated_regions, result.area, result.source_image)
            except Exception as error:
                self.status_var.set(f"Перевод получен, но оверлей не показан: {error}")
                self._append_log(f"Ошибка оверлея: {error}")
                return
            self._append_log(
                f"Оверлей {(perf_counter() - overlay_started) * 1000:.0f} мс; "
                f"от запроса до показа {(perf_counter() - self._capture_started) * 1000:.0f} мс."
            )
            if result.cache_hit:
                self.status_var.set(
                    f"Показано из кэша: {len(result.translated_regions)} блок(а/ов) · "
                    f"OCR и перевод повторно не выполнялись."
                )
            else:
                self.status_var.set(
                    f"Готово: {len(result.translated_regions)} блок(а/ов) · "
                    f"OCR {result.ocr_ms:.0f} мс · перевод {result.translate_ms:.0f} мс · "
                    f"захват {result.capture_ms:.0f} мс"
                )
            self._append_log(
                f"Захват #{capture_id}: OCR {result.ocr_ms:.0f} мс; "
                f"перевод {result.translate_ms:.0f} мс; "
                f"показано блоков: {len(result.translated_regions)}"
                + (" (кэш кадра)." if result.cache_hit else ".")
            )
        elif result.recognized_count == 0:
            self.status_var.set("Текст не найден в выбранной области.")
            self._append_log("OCR не распознал текст.")
        elif result.filtered_count == result.recognized_count and not result.errors:
            self.status_var.set(
                "В выбранной области только числа, даты или кодоподобные строки."
            )
            self._append_log(f"Пропущено фильтром блоков: {result.filtered_count}.")
        elif result.errors:
            self.status_var.set(
                "Не удалось перевести подходящий текст. Подробности — в журнале."
            )
        else:
            self.status_var.set(
                "Перевод не изменился или не был получен; оверлей не показан."
            )
            self._append_log("Неотличимые от оригинала результаты не отображены.")

    def run(self) -> None:
        self.root.mainloop()
        if not self._shutting_down:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._append_log("Завершение приложения.")
        self._drain_ui_logs()
        with self._runtime_publish_lock:
            self._shutting_down = True
            # A setup result can be queued just before the window closes.
            # It has not yet become self._runtime and must also be released.
            while True:
                try:
                    event = self._events.get_nowait()
                except Empty:
                    break
                if event[0] == "runtime_ready":
                    self._close_runtime(event[1])
        self._capture_cancel.set()
        self._active = False
        self._remove_hotkeys()
        self._save_settings()
        pipeline_logger = getattr(self, "_pipeline_logger", None)
        app_logger = getattr(self, "_logger", None)
        if pipeline_logger is not None and app_logger is not None:
            for handler in list(pipeline_logger.handlers):
                if handler in app_logger.handlers:
                    pipeline_logger.removeHandler(handler)
        for logger, handler_name in (
            (getattr(self, "_logger", None), "_ui_log_handler"),
            (getattr(self, "_pipeline_logger", None), "_pipeline_log_handler"),
            (logging.getLogger(), "_dependency_log_handler"),
        ):
            handler = getattr(self, handler_name, None)
            if logger is not None and handler is not None:
                try:
                    logger.removeHandler(handler)
                    handler.close()
                except Exception:
                    pass
        try:
            if self._area_selector is not None:
                self._area_selector.cancel()
        except Exception:
            pass
        try:
            self.overlay.close()
        except Exception:
            pass

        file_handler = getattr(self, "_session_file_handler", None)
        if file_handler is not None and app_logger is not None:
            app_logger.removeHandler(file_handler)
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            self._session_file_handler = None

        runtime = self._runtime

        def release_resources() -> None:
            # Run after the capture, on the thread that owns MSS resources.
            if runtime is not None:
                self._close_runtime(runtime)
            try:
                self.screen_capturer.close()
            except Exception:
                pass

        self._capture_executor.submit(release_resources)
        self._capture_executor.shutdown(wait=False)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    @staticmethod
    def _close_runtime(runtime: RuntimeBundle) -> None:
        try:
            runtime.ocr_worker.close(timeout=1.0)
        except Exception:
            pass
        try:
            runtime.translator.close()
        except Exception:
            pass
