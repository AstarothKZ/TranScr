"""Translation module public API."""

__all__ = ["Translator"]


def __getattr__(name):
    # Loading the provider catalog for settings must not load HTTP clients,
    # Tk dialogs or .env files. Keep the public import backwards compatible.
    if name == "Translator":
        from .translator import Translator
        return Translator
    raise AttributeError(name)
