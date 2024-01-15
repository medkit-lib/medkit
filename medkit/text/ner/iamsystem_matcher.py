from __future__ import annotations

__all__ = ["IAMSystemMatcher", "MedkitKeyword"]

from typing import Any, Callable, Optional, Sequence, runtime_checkable

from iamsystem import Annotation as IS_Annotation
from iamsystem import IEntity as IS_IEntity
from iamsystem import IKeyword as IS_IKeyword
from iamsystem import Matcher as IS_Matcher
from typing_extensions import Protocol

from medkit.core.text import (
    Entity,
    EntityNormAttribute,
    NEROperation,
    Segment,
    span_utils,
)


@runtime_checkable
class SupportKBName(IS_IEntity, Protocol):
    kb_name: str | None


@runtime_checkable
class SupportEntLabel(IS_IKeyword, Protocol):
    ent_label: str | None


class MedkitKeyword:
    """A recommended iamsystem's IEntity implementation.

    This class is implemented to allow user to define one of both values of `kb_id`
    or `kb_name` with its iamsystem keyword.
    The entity label may be also provided if the user wants to define a category for
    the searched keyword (e.g., "drug" label for "Vicodin" keyword)


    Also implements :class:`~.SupportEntLabel`, :class:`~.SupportKBName` protocols
    """

    def __init__(
        self,
        label: str,  # String to search in text
        kb_id: Any | None,
        kb_name: str | None,
        ent_label: str | None,  # Output label for the detected entity
    ):
        self.label = label
        self.kb_id = kb_id
        self.kb_name = kb_name
        self.ent_label = ent_label


LabelProvider = Callable[[Sequence[IS_IKeyword]], Optional[str]]


class DefaultLabelProvider:
    """Default entity label provider"""

    @staticmethod
    def __call__(keywords: Sequence[IS_IKeyword]) -> str | None:
        """Uses the first keyword which implements`SupportEntLabel` protocol and returns
        `ent_label`. Otherwise, returns None.
        """
        for kw in keywords:
            if isinstance(kw, SupportEntLabel) and kw.ent_label is not None:
                return kw.ent_label
        return None


class IAMSystemMatcher(NEROperation):
    """Entity annotator and linker based on iamsystem library"""

    def __init__(
        self,
        matcher: IS_Matcher,
        label_provider: LabelProvider | None = None,
        attrs_to_copy: list[str] | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Instantiate the operation supporting the iamsystem matcher

        Parameters
        ----------
        matcher : IS_Matcher
            IAM system Matcher
        label_provider : LabelProvider, optional
            Callable providing the output label to set for detected entity.
            As iamsystem matcher may return several keywords for an annotation,
            we have to know how to provide only one entity label whatever the
            number of matched keywords.
            In medkit, normalization attributes are used for representing detected
            keywords.
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the input segment
            to the created entity. Useful for propagating context attributes
            (negation, antecedent, etc).
        name : str, optional
            Name describing the matcher (defaults to the class name)
        uid : str, optional
            Identifier of the operation
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.matcher = matcher
        self.label_provider = label_provider or DefaultLabelProvider()
        self.attrs_to_copy = attrs_to_copy

    def run(self, segments: list[Segment]) -> list[Entity]:
        return [
            self._create_entity_from_iamsystem_ann(ann, segment)
            for segment in segments
            for ann in self.matcher.annot_text(segment.text)
        ]

    def _create_entity_from_iamsystem_ann(self, ann: IS_Annotation, segment: Segment):
        ranges = []
        positions = []
        pos = 0
        for token in ann.tokens:
            ranges.append((token.start, token.end))
            pos += token.end - token.start
            positions.append(pos)
        positions.pop()

        # Convert list of ann tokens spans to medkit spans
        text, spans = span_utils.extract(segment.text, segment.spans, ranges)
        inserts = [" " for i in positions]
        text, spans = span_utils.insert(text, spans, positions, inserts)

        tokens_algos = [dict(token.__dict__, algos=algos) for token, algos in ann.get_tokens_algos()]

        metadata = {
            "tokens_algos": tokens_algos,
        }
        ent_label = self.label_provider(ann.keywords)
        if ent_label is None:
            ent_label = ann.keywords[0].label
        entity = Entity(
            label=ent_label,
            text=text,
            spans=spans,
            metadata=metadata,
        )

        # Copy inherited attributes
        for label in self.attrs_to_copy:
            for attr in segment.attrs.get(label=label):
                copied_attr = attr.copy()
                entity.attrs.add(copied_attr)
                # handle provenance
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(copied_attr, self.description, [attr])

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])

        # create normalization attributes
        for kw in ann.keywords:
            kb_id = None
            if isinstance(kw, IS_IEntity):
                kb_id = kw.kb_id
            kb_name = None
            if isinstance(kw, SupportKBName) and kw.kb_name is not None:
                kb_name = kw.kb_name
            term = kw.label
            if any([kb_name, kb_id]):
                norm_attr = EntityNormAttribute(kb_name=kb_name, kb_id=kb_id, term=term)
                norm_attr = entity.attrs.add(norm_attr)

                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(norm_attr, self.description, source_data_items=[segment])

        return entity
