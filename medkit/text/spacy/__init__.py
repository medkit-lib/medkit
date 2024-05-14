from medkit._import import import_optional

_ = import_optional("spacy")


from medkit.text.spacy.doc_pipeline import SpacyDocPipeline  # noqa: E402
from medkit.text.spacy.pipeline import SpacyPipeline  # noqa: E402

__all__ = ["SpacyDocPipeline", "SpacyPipeline"]


try:
    from medkit.text.spacy.edsnlp import EDSNLPDocPipeline, EDSNLPPipeline

    __all__ += ["EDSNLPDocPipeline", "EDSNLPPipeline"]
except ModuleNotFoundError:
    pass
