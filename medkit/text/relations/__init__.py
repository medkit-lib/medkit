__all__ = []

try:
    from medkit.text.relations.syntactic_relation_extractor import SyntacticRelationExtractor

    __all__ += ["SyntacticRelationExtractor"]
except ModuleNotFoundError:
    pass
