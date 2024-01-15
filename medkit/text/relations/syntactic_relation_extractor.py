"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[syntactic-relation-extractor]`.
"""
from __future__ import annotations

__all__ = ["SyntacticRelationExtractor"]

import logging
from typing import TYPE_CHECKING

import spacy

from medkit.core.operation import DocOperation
from medkit.core.text import Relation, TextDocument
from medkit.text.spacy import spacy_utils

if TYPE_CHECKING:
    from pathlib import Path

    from spacy.tokens import Doc
    from spacy.tokens import Span as SpacySpan

logger = logging.getLogger(__name__)


class SyntacticRelationExtractor(DocOperation):
    """Extractor of syntactic relations between entities in a TextDocument.
    The relation relies on the dependency parser from a spacy pipeline.
    A transition-based dependency parser defines a dependency tag for each
    token (word) in a document. This relation extractor uses syntactic neighbours
    of the words of an entity to determine whether a dependency exists
    between the entities.

    Each TextDocument is converted to a spacy doc with the entities of interest.
    The labels of entities to be used as sources and targets of the relation
    are provided by the user, but it is also possible to not restrict the labels
    of sources and/or target entities. If neither the source label nor the
    target labels are provided, the 'SyntacticRelationExtractor' will detect
    relations among all entities in the document, and the order of the relation
    will be the syntactic order.
    """

    _DEFAULT_NAME_SPACY_MODEL = "fr_core_news_sm"
    _DEFAULT_LABEL = "has_syntactic_rel"

    def __init__(
        self,
        name_spacy_model: str | Path = _DEFAULT_NAME_SPACY_MODEL,
        relation_label: str = _DEFAULT_LABEL,
        entities_source: list[str] | None = None,
        entities_target: list[str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Initialize the syntactic relation extractor

        Parameters
        ----------
        name_spacy_model : str, optional
            Name or path of a spacy pipeline to load, it should include a
            syntactic dependency parser. To obtain consistent results,
            the spacy model should have the same language as the documents
            in which relations should be found.
        relation_label : str, optional
            Label of identified relations
        entities_source : list of str, optional
            Labels of medkit entities to use as source of the relation.
            If `None`, any entity can be used as source.
        entities_target : list of str, optional
            Labels of medkit entities to use as target of the relation.
            If `None`, any entity can be used as target.
        name : str, optional
            Name describing the relation extractor (defaults to the class name)
        uid : str, optional
            Identifier of the relation extractor

        Raises
        ------
        ValueError
            If the spacy model defined by `name_spacy_model` does not parse a document
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        # load nlp object and validate it
        nlp = spacy.load(name_spacy_model, exclude=["tagger", "ner", "lemmatizer"])
        if not nlp("X").has_annotation("DEP"):
            msg = (
                f"Model `{name_spacy_model}` does not add syntax attributes"
                " to documents and cannot be use with SyntacticRelationExtractor."
            )
            raise ValueError(msg)

        self._nlp = nlp
        self.name_spacy_model = name_spacy_model
        self.entities_source = entities_source
        self.entities_target = entities_target
        self.relation_label = relation_label
        # entities transferred to the equivalent spacy doc
        if self.entities_source and self.entities_target:
            self._entities_labels = self.entities_source + self.entities_target
        else:
            self._entities_labels = None

    def run(self, documents: list[TextDocument]):
        """Add relations to each document from `documents`

        Parameters
        ----------
        documents : list of TextDocument
            List of text documents in which relations are to be found

        """
        for medkit_doc in documents:
            # build spacy doc using selected entities
            spacy_doc = spacy_utils.build_spacy_doc_from_medkit_doc(
                nlp=self._nlp,
                medkit_doc=medkit_doc,
                labels_anns=self._entities_labels,
                attrs=[],
                include_medkit_info=True,
            )
            # apply nlp spacy to include dependency tag
            spacy_doc = self._nlp(spacy_doc)
            relations = self._find_syntactic_relations(spacy_doc)
            self._add_relations_to_document(medkit_doc, relations)

    def _find_syntactic_relations(self, spacy_doc: Doc):
        """Find syntactic relations from entities present in the same sentence.
        For each dependency found, a new relation is created.

        Parameters
        ----------
        spacy_doc : Doc
            A spacy doc with medkit entities converted in spacy entities

        Returns
        -------
        Relation
            The Relation object representing the spacy relation
        """
        relations = []
        for sentence in spacy_doc.sents:
            ents = sentence.ents
            # find a binary relation between e1 and e2
            for idx, e1 in enumerate(ents[:-1]):
                e2 = ents[idx + 1]
                right_child_tokens_of_e1 = [token.i for token in e1.rights]
                left_child_tokens_of_e2 = [token.i for token in e2.lefts]
                if any(token.i in right_child_tokens_of_e1 for token in e2):
                    # a relation left to right exist, e1 is the syntactic source
                    # of the relation, check if source or target is predefined
                    source, target = self._define_source_target(e1, e2)
                    relation = self._create_relation(
                        source=source,
                        target=target,
                        metadata={"dep_tag": e2.root.dep_, "dep_direction": "left_to_right"},
                    )
                    if relation is not None:
                        relations.append(relation)

                if any(token.i in left_child_tokens_of_e2 for token in e1):
                    # a relation right to left exist, e2 is the syntactic source
                    # of the relation, check if source or target is predefined
                    source, target = self._define_source_target(e2, e1)
                    relation = self._create_relation(
                        source=source,
                        target=target,
                        metadata={"dep_tag": e1.root.dep_, "dep_direction": "right_to_left"},
                    )
                    if relation is not None:
                        relations.append(relation)

        return relations

    def _define_source_target(self, source: SpacySpan, target: SpacySpan):
        # change whether origin or target is predefined by the user
        if (self.entities_target and source.label_ in self.entities_target) or (
            self.entities_source and target.label_ in self.entities_source
        ):
            return target, source
        return source, target

    def _create_relation(self, source: SpacySpan, target: SpacySpan, metadata: dict[str, str]) -> Relation | None:
        """Parse the spacy relation content into a Relation object.

        Parameters
        ----------
        source : SpacySpan
            Spacy entity source of the syntactic relation
        target : SpacySpan
            Spacy entity target of the syntactic relation
        metadata : dict of str to str
            Additional information of the relation

        Returns
        -------
        Relation, optional
            The Relation object representing the spacy relation

        """
        # ignore the relation because it has no entities of interest
        if self.entities_source and source.label_ not in self.entities_source:
            return None
        if self.entities_target and target.label_ not in self.entities_target:
            return None

        attribute_medkit_id = spacy_utils._ATTR_MEDKIT_ID
        source_id = source._.get(attribute_medkit_id)
        target_id = target._.get(attribute_medkit_id)

        if source_id is None or target_id is None:
            logging.warning(
                "Can't create a medkit Relation between `%s` and"
                " `%s`. Source or target entity has not been detected by"
                " medkit but spacy pipeline, and it is not supported by this module.",
                source.text,
                target.text,
            )
            return None

        return Relation(
            source_id=source_id,
            target_id=target_id,
            label=self.relation_label,
            metadata=metadata,
        )

    def _add_relations_to_document(self, medkit_doc: TextDocument, relations: list[Relation]):
        for relation in relations:
            medkit_doc.anns.add(relation)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    relation,
                    self.description,
                    source_data_items=[medkit_doc.raw_segment],
                )
