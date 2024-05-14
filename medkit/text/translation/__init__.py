__all__ = []

try:
    from medkit.text.translation.hf_translator import HFTranslator

    __all__ += ["HFTranslator"]
except ModuleNotFoundError:
    pass
