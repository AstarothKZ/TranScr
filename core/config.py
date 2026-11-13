"""JSON settings storage and validation for the application."""

import json
import os
from pathlib import Path
from typing import Any

from modules.translation.catalog import PROVIDER_LABELS

PROVIDERS = frozenset(PROVIDER_LABELS)


class AppConfig:
    """Persist lightweight preferences; API secrets are stored separately in .env."""

    DEFAULTS = {
        "ocr": "windows",
        "provider": "google",
        "hybrid_enabled": False,
        "fallback_provider": "argos",
        "start_hotkey": "alt+t",
        "quick_area_hotkey": "alt+q",
        "dismiss_mode": "any",
        "dismiss_hotkey": "esc",
        "area": None,
        "save_debug_screenshots": False,
        "repair_incomplete": False,
    }

    def __init__(self, project_root: Path) -> None:
        if os.name == "nt":
            base = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "TranScr"
            self.path = base / "settings.json"
        else:
            self.path = Path(project_root) / ".transcr_settings.json"

    def load(self) -> dict[str, Any]:
        result = dict(self.DEFAULTS)
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in result:
                    if key in loaded:
                        result[key] = loaded[key]
                # Migrate the old single "hybrid" provider choice to a primary
                # provider plus an independently configurable fallback.
                if loaded.get("provider") == "hybrid":
                    result["provider"] = "google"
                    result["hybrid_enabled"] = True
                    result["fallback_provider"] = "argos"
        except (OSError, ValueError, TypeError):
            pass

        if not isinstance(result["ocr"], str) or result["ocr"] not in {
            "windows",
            "paddle",
        }:
            result["ocr"] = self.DEFAULTS["ocr"]
        if (
            not isinstance(result["provider"], str)
            or result["provider"] not in PROVIDERS
        ):
            result["provider"] = self.DEFAULTS["provider"]
        if (
            not isinstance(result["fallback_provider"], str)
            or result["fallback_provider"] not in PROVIDERS
        ):
            result["fallback_provider"] = (
                "google" if result["provider"] != "google" else "argos"
            )
        if result["fallback_provider"] == result["provider"]:
            result["fallback_provider"] = (
                "argos" if result["provider"] != "argos" else "google"
            )
        if not isinstance(result["hybrid_enabled"], bool):
            result["hybrid_enabled"] = False
        if not isinstance(result["dismiss_mode"], str) or result[
            "dismiss_mode"
        ] not in {"any", "hotkey"}:
            result["dismiss_mode"] = "any"
        for key, default in (
            ("start_hotkey", "alt+t"),
            ("dismiss_hotkey", "esc"),
            ("quick_area_hotkey", "alt+q"),
        ):
            value = result.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 64:
                result[key] = default
            else:
                result[key] = value.strip().lower()
        if not isinstance(result["save_debug_screenshots"], bool):
            result["save_debug_screenshots"] = False
        if not isinstance(result["repair_incomplete"], bool):
            result["repair_incomplete"] = False
        if not self._valid_area(result["area"]):
            result["area"] = None
        return result

    def save(self, settings: dict[str, Any]) -> None:
        payload = dict(self.DEFAULTS)
        for key in payload:
            if key in settings:
                payload[key] = settings[key]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            # A read-only / portable project should still run without persistence.
            pass

    @staticmethod
    def _valid_area(value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return False
        return (
            all(type(item) is int for item in value)
            and value[2] >= 20
            and value[3] >= 12
        )
