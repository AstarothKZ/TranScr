"""Settings and live diagnostics tabs for TranScr's Tkinter interface."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Mapping
from tkinter import ttk
from typing import Any


class SettingsWindow:
    """Present settings and logs; all application actions are injected as callbacks."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        saved_settings: Mapping[str, Any],
        ocr_labels: Mapping[str, str],
        ocr_values: Mapping[str, str],
        provider_labels: Mapping[str, str],
        provider_values: Mapping[str, str],
        initial_area_text: str,
        callbacks: Mapping[str, Callable[[], None]],
    ) -> None:
        self.root = root
        self._callbacks = callbacks
        self.ocr_labels = dict(ocr_labels)
        self.ocr_values = dict(ocr_values)
        self.provider_labels = dict(provider_labels)
        self.provider_values = dict(provider_values)

        # Settings-only window: no product title banner or descriptive text.
        self.root.title("Настройки")
        self.root.geometry("650x780")
        self.root.minsize(610, 720)

        self.ocr_var = tk.StringVar(
            value=self.ocr_values.get(
                saved_settings.get("ocr", "windows"), "Windows OCR"
            )
        )
        provider = saved_settings.get("provider", "mymemory")
        self.provider_var = tk.StringVar(
            value=self.provider_labels.get(provider, self.provider_labels["mymemory"])
        )
        fallback = saved_settings.get("fallback_provider", "argos")
        self.fallback_provider_var = tk.StringVar(
            value=self.provider_labels.get(fallback, self.provider_labels["argos"])
        )
        self.hybrid_var = tk.BooleanVar(
            value=bool(saved_settings.get("hybrid_enabled", False))
        )
        self.start_hotkey_var = tk.StringVar(
            value=saved_settings.get("start_hotkey", "alt+t")
        )
        self.quick_area_hotkey_var = tk.StringVar(
            value=saved_settings.get("quick_area_hotkey", "alt+q")
        )
        self.dismiss_mode_var = tk.StringVar(
            value=saved_settings.get("dismiss_mode", "any")
        )
        self.dismiss_hotkey_var = tk.StringVar(
            value=saved_settings.get("dismiss_hotkey", "esc")
        )
        self.area_var = tk.StringVar(value=initial_area_text)
        self.save_debug_var = tk.BooleanVar(
            value=bool(saved_settings.get("save_debug_screenshots", False))
        )
        self.status_var = tk.StringVar(
            value="Готово. Проверьте настройки и запустите перевод."
        )

        self._configuration_enabled = True
        self._build_widgets()

    def _build_widgets(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        # Reserve status space before the expandable notebook can consume it.
        self.status_label = ttk.Label(
            container, textvariable=self.status_var, wraplength=590,
            justify="left", foreground="#245c35", padding=(4, 8),
        )
        self.status_label.pack(side="bottom", fill="x")
        container.bind('<Configure>', lambda event: self.status_label.configure(
            wraplength=max(200, event.width-28)))

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)
        self.settings_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.settings_tab, text="Настройки")
        self.notebook.add(self.log_tab, text="Журнал")

        outer = ttk.Frame(self.settings_tab, padding=12)
        outer.pack(fill="both", expand=True)

        engine_frame = ttk.LabelFrame(outer, text="Обработка и перевод", padding=10)
        engine_frame.pack(fill="x")
        engine_frame.columnconfigure(0, weight=1)

        ttk.Label(engine_frame, text="Распознавание текста (OCR)").grid(
            row=0, column=0, sticky="w"
        )
        self.ocr_combo = ttk.Combobox(
            engine_frame,
            textvariable=self.ocr_var,
            values=list(self.ocr_labels),
            state="readonly",
        )
        self.ocr_combo.grid(row=1, column=0, sticky="ew", pady=(3, 9))

        ttk.Label(engine_frame, text="Переводчик / основная модель").grid(
            row=2, column=0, sticky="w"
        )
        self.provider_combo = ttk.Combobox(
            engine_frame,
            textvariable=self.provider_var,
            values=list(self.provider_values),
            state="readonly",
        )
        self.provider_combo.grid(row=3, column=0, sticky="ew", pady=(3, 6))
        self.provider_combo.bind(
            "<<ComboboxSelected>>", self._on_primary_provider_changed
        )

        hybrid_row = ttk.Frame(engine_frame)
        hybrid_row.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.hybrid_button = ttk.Button(
            hybrid_row, text="Включить гибридный режим", command=self.toggle_hybrid
        )
        self.hybrid_button.pack(side="left", ipadx=4, ipady=2)

        fallback_row = ttk.Frame(engine_frame)
        fallback_row.grid(row=5, column=0, sticky="ew", pady=(0, 3))
        ttk.Label(fallback_row, text="Резервный переводчик").pack(side="left")
        self.fallback_combo = ttk.Combobox(
            fallback_row,
            textvariable=self.fallback_provider_var,
            values=list(self.provider_values),
            state="readonly",
            width=32,
        )
        self.fallback_combo.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.fallback_combo.bind("<<ComboboxSelected>>", self._on_fallback_changed)
        self._update_hybrid_controls()

        hotkey_frame = ttk.LabelFrame(outer, text="Горячие клавиши", padding=10)
        hotkey_frame.pack(fill="x", pady=(9, 0))
        hotkey_frame.columnconfigure(1, weight=1)

        ttk.Label(hotkey_frame, text="Запуск перевода").grid(
            row=0, column=0, sticky="w"
        )
        self.start_hotkey_entry = ttk.Entry(
            hotkey_frame, textvariable=self.start_hotkey_var, width=22
        )
        self.start_hotkey_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Label(
            hotkey_frame, text="Например: alt+t, ctrl+shift+x, f8", foreground="#666666"
        ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(2, 7))

        ttk.Label(hotkey_frame, text="Сброс перевода").grid(
            row=2, column=0, sticky="nw"
        )
        dismiss_options = ttk.Frame(hotkey_frame)
        dismiss_options.grid(row=2, column=1, sticky="ew", padx=(10, 0))
        self.dismiss_any_radio = ttk.Radiobutton(
            dismiss_options,
            text="Любая клавиша",
            value="any",
            variable=self.dismiss_mode_var,
            command=self._on_dismiss_mode_changed,
        )
        self.dismiss_any_radio.pack(side="left")
        self.dismiss_custom_radio = ttk.Radiobutton(
            dismiss_options,
            text="Сочетание",
            value="hotkey",
            variable=self.dismiss_mode_var,
            command=self._on_dismiss_mode_changed,
        )
        self.dismiss_custom_radio.pack(side="left", padx=(12, 0))
        self.dismiss_hotkey_entry = ttk.Entry(
            hotkey_frame, textvariable=self.dismiss_hotkey_var, width=22
        )
        self.dismiss_hotkey_entry.grid(
            row=3, column=1, sticky="ew", padx=(10, 0), pady=(5, 0)
        )
        ttk.Label(
            hotkey_frame,
            text="Пример: esc или ctrl+shift+backspace",
            foreground="#666666",
        ).grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(2, 0))
        self._update_dismiss_controls()

        ttk.Label(hotkey_frame, text="Выделить и перевести").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        self.quick_area_hotkey_entry = ttk.Entry(
            hotkey_frame, textvariable=self.quick_area_hotkey_var, width=22
        )
        self.quick_area_hotkey_entry.grid(
            row=5, column=1, sticky="ew", padx=(10, 0), pady=(8, 0)
        )
        self.quick_area_hotkey_entry.bind("<FocusOut>", lambda _event: self._save())

        area_frame = ttk.LabelFrame(outer, text="Область захвата", padding=10)
        area_frame.pack(fill="x", pady=(9, 0))
        ttk.Label(area_frame, textvariable=self.area_var, wraplength=570).pack(
            anchor="w"
        )
        area_buttons = ttk.Frame(area_frame)
        area_buttons.pack(fill="x", pady=(7, 0))
        self.select_area_button = ttk.Button(
            area_buttons, text="Выбрать мышью…", command=self._callbacks["select_area"]
        )
        self.select_area_button.pack(side="left", ipadx=3, ipady=1)
        self.full_screen_button = ttk.Button(
            area_buttons,
            text="Весь экран",
            command=self._callbacks["select_full_screen"],
        )
        self.full_screen_button.pack(side="left", padx=(8, 0), ipadx=3, ipady=1)
        self.save_debug_checkbutton = ttk.Checkbutton(
            area_frame,
            text="Сохранять скриншоты для отладки",
            variable=self.save_debug_var,
            command=self._save,
        )
        self.save_debug_checkbutton.pack(anchor="w", pady=(7, 0))

        buttons = ttk.Frame(container)
        # Allocate actions before the expanding notebook, just like status.
        buttons.pack(side="bottom", fill="x", pady=(8, 4), before=self.notebook)
        for column in range(3):
            buttons.columnconfigure(column, weight=1, uniform="action_buttons")
        # Classic Tk buttons are used for the three main actions. On some Windows
        # ttk themes these controls were rendered as extremely thin blank strips.
        # Explicit font, padding, colours and width keep their labels visible.
        action_font = ("Segoe UI", 9, "bold")
        self.start_button = tk.Button(
            buttons,
            text="Запустить переводчик",
            command=self._callbacks["start"],
            font=action_font,
            width=19,
            height=2,
            padx=6,
            pady=4,
            bg="#d9ead3",
            fg="#173b1b",
            activebackground="#c5e1bc",
            disabledforeground="#666666",
            relief="raised",
            borderwidth=1,
            cursor="hand2",
            takefocus=True,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.capture_button = tk.Button(
            buttons,
            text="Перевести сейчас",
            command=self._callbacks["capture"],
            state="disabled",
            font=action_font,
            width=17,
            height=2,
            padx=6,
            pady=4,
            bg="#e8edf3",
            fg="#1f2937",
            activebackground="#d4dee9",
            disabledforeground="#777777",
            relief="raised",
            borderwidth=1,
            cursor="hand2",
            takefocus=True,
        )
        self.capture_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.stop_button = tk.Button(
            buttons,
            text="Остановить",
            command=self._callbacks["stop"],
            state="disabled",
            font=action_font,
            width=12,
            height=2,
            padx=6,
            pady=4,
            bg="#f4cccc",
            fg="#5b1111",
            activebackground="#edaaaa",
            disabledforeground="#777777",
            relief="raised",
            borderwidth=1,
            cursor="hand2",
            takefocus=True,
        )
        self.stop_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.start_hotkey_entry.bind("<FocusOut>", lambda _event: self._save())
        self.dismiss_hotkey_entry.bind("<FocusOut>", lambda _event: self._save())
        self.ocr_combo.bind("<<ComboboxSelected>>", lambda _event: self._save())

        self._build_log_tab()

    def _build_log_tab(self) -> None:
        self.log_tab.rowconfigure(0, weight=1)
        self.log_tab.columnconfigure(0, weight=1)
        log_frame = ttk.Frame(self.log_tab)
        log_frame.grid(row=0, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            height=20,
            font=("Consolas", 9),
            background="#17191c",
            foreground="#e7e7e7",
            insertbackground="#ffffff",
            relief="flat",
            padx=8,
            pady=8,
        )
        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.tag_configure("INFO", foreground="#e7e7e7")
        self.log_text.tag_configure("WARNING", foreground="#ffcc66")
        self.log_text.tag_configure("ERROR", foreground="#ff7777")
        self.log_text.tag_configure("DEBUG", foreground="#9dc7ff")

        controls = ttk.Frame(self.log_tab)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Очистить журнал", command=self.clear_log).pack(
            side="left"
        )
        self.log_file_label = ttk.Label(
            controls, text="Файл журнала недоступен", foreground="#666666", wraplength=400
        )
        self.log_file_label.pack(side="right")

    def append_log(self, message: str, level: int = 20) -> None:
        """Append a log event on Tk's main thread and keep memory use bounded."""
        if not hasattr(self, "log_text"):
            return
        if level >= 40:
            tag = "ERROR"
        elif level >= 30:
            tag = "WARNING"
        elif level <= 10:
            tag = "DEBUG"
        else:
            tag = "INFO"
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{message}\n", (tag,))
            # Avoid a long-running session growing the UI text widget indefinitely.
            line_count = int(self.log_text.index("end-1c").split(".")[0])
            if line_count > 2000:
                self.log_text.delete("1.0", f"{line_count - 1800}.0")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except tk.TclError:
            pass

    def clear_log(self) -> None:
        try:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")
        except tk.TclError:
            pass

    def _on_primary_provider_changed(self, _event=None) -> None:
        if self.provider_var.get() == self.fallback_provider_var.get():
            primary = self.provider_values.get(self.provider_var.get(), "mymemory")
            replacement = "argos" if primary != "argos" else "mymemory"
            self.fallback_provider_var.set(self.provider_labels[replacement])
        self._save()

    def _on_fallback_changed(self, _event=None) -> None:
        if self.provider_var.get() == self.fallback_provider_var.get():
            primary = self.provider_values.get(self.provider_var.get(), "mymemory")
            replacement = "argos" if primary != "argos" else "mymemory"
            self.fallback_provider_var.set(self.provider_labels[replacement])
        self._save()

    def toggle_hybrid(self) -> None:
        self.hybrid_var.set(not self.hybrid_var.get())
        self._update_hybrid_controls()
        self._save()

    def _update_hybrid_controls(self) -> None:
        hybrid_enabled = self.hybrid_var.get()
        self.hybrid_button.configure(
            text="Гибридный режим: ВКЛ"
            if hybrid_enabled
            else "Включить гибридный режим",
            state="normal" if self._configuration_enabled else "disabled",
        )
        can_edit_fallback = self._configuration_enabled and hybrid_enabled
        self.fallback_combo.configure(
            state="readonly" if can_edit_fallback else "disabled"
        )

    def _on_dismiss_mode_changed(self) -> None:
        self._update_dismiss_controls()
        self._save()

    def _update_dismiss_controls(self) -> None:
        state = "normal" if self._configuration_enabled else "disabled"
        self.dismiss_any_radio.configure(state=state)
        self.dismiss_custom_radio.configure(state=state)
        self.dismiss_hotkey_entry.configure(
            state="normal"
            if self._configuration_enabled and self.dismiss_mode_var.get() == "hotkey"
            else "disabled"
        )

    def set_configuration_enabled(self, enabled: bool) -> None:
        """Lock mutable settings while components are starting or running."""
        self._configuration_enabled = bool(enabled)
        widget_state = "readonly" if enabled else "disabled"
        self.ocr_combo.configure(state=widget_state)
        self.provider_combo.configure(state=widget_state)
        self.start_hotkey_entry.configure(state="normal" if enabled else "disabled")
        self.quick_area_hotkey_entry.configure(
            state="normal" if enabled else "disabled"
        )
        self.save_debug_checkbutton.configure(state="normal" if enabled else "disabled")
        self._update_hybrid_controls()
        self._update_dismiss_controls()

    def _save(self) -> None:
        callback = self._callbacks.get("save_settings")
        if callback is not None:
            callback()
