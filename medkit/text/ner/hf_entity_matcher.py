"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[hf-entity-matcher]`.
"""
from __future__ import annotations

__all__ = ["HFEntityMatcher"]

from typing import TYPE_CHECKING, Iterator

import transformers
from transformers import TokenClassificationPipeline
from typing_extensions import Literal

from medkit.core import Attribute
from medkit.core.text import Entity, NEROperation, Segment, span_utils
from medkit.text.ner.hf_entity_matcher_trainable import HFEntityMatcherTrainable
from medkit.tools import hf_utils

if TYPE_CHECKING:
    from pathlib import Path


class HFEntityMatcher(NEROperation):
    """Entity matcher based on HuggingFace transformers model

    Any token classification model from the HuggingFace hub can be used
    (for instance "samrawal/bert-base-uncased_clinical-ner").
    """

    def __init__(
        self,
        model: str | Path,
        aggregation_strategy: Literal["none", "simple", "first", "average", "max"] = "max",
        attrs_to_copy: list[str] | None = None,
        device: int = -1,
        batch_size: int = 1,
        hf_auth_token: str | None = None,
        cache_dir: str | Path | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        model : str or Path
            Name (on the HuggingFace models hub) or path of the NER model. Must be a model compatible
            with the `TokenClassification` transformers class.
        aggregation_strategy : str, default="max"
            Strategy to fuse tokens based on the model prediction, passed to `TokenClassificationPipeline`.
            Defaults to "max", cf https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy
            for details
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the input segment
            to the created entity. Useful for propagating context attributes
            (negation, antecendent, etc).
        device : int, default=-1
            Device to use for the transformer model. Follows the HuggingFace convention
            (-1 for "cpu" and device number for gpu, for instance 0 for "cuda:0").
        batch_size : int, default=1
            Number of segments in batches processed by the transformer model.
        hf_auth_token : str, optional
            HuggingFace Authentication token (to access private models on the
            hub)
        cache_dir : str or Path, optional
            Directory where to store downloaded models. If not set, the default
            HuggingFace cache dir is used.
        name : str, optional
            Name describing the matcher (defaults to the class name).
        uid : str, optional
            Identifier of the matcher.
        """
        # Pass all arguments to super (remove self and confidential hf_auth_token)
        init_args = locals()
        init_args.pop("self")
        init_args.pop("hf_auth_token")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.model = model
        self.attrs_to_copy = attrs_to_copy

        valid_model = hf_utils.check_model_for_task_hf(self.model, "token-classification", hf_auth_token=hf_auth_token)

        if not valid_model:
            msg = (
                f"Model {self.model} is not associated to a"
                " token-classification/ner task and cannot be used with"
                " HFEntityMatcher"
            )
            raise ValueError(msg)

        self._pipeline = transformers.pipeline(
            task="token-classification",
            model=self.model,
            aggregation_strategy=aggregation_strategy,
            pipeline_class=TokenClassificationPipeline,
            device=device,
            batch_size=batch_size,
            token=hf_auth_token,
            model_kwargs={"cache_dir": cache_dir},
        )

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Return entities for each match in `segments`.

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for matches.

        Returns
        -------
        list of Entity
            Entities found in `segments`.
        """
        # get an iterator to all matches, grouped by segment
        all_matches = self._pipeline(x.text for x in segments)
        # build entities from matches
        return [
            entity
            for matches, segment in zip(all_matches, segments)
            for entity in self._matches_to_entities(matches, segment)
        ]

    def _matches_to_entities(self, matches: list[dict], segment: Segment) -> Iterator[Entity]:
        for match in matches:
            text, spans = span_utils.extract(segment.text, segment.spans, [(match["start"], match["end"])])

            entity = Entity(
                label=match["entity_group"],
                text=text,
                spans=spans,
            )

            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    entity.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            score_attr = Attribute(
                label="score",
                value=float(match["score"]),
                metadata={"model": self.model},
            )
            entity.attrs.add(score_attr)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])
                self._prov_tracer.add_prov(score_attr, self.description, source_data_items=[segment])

            yield entity

    @staticmethod
    def make_trainable(
        model_name_or_path: str | Path,
        labels: list[str],
        tagging_scheme: Literal["bilou", "iob2"],
        tag_subtokens: bool = False,
        tokenizer_max_length: int | None = None,
        hf_auth_token: str | None = None,
        device: int = -1,
    ):
        """Return the trainable component of the operation.
        This component can be trained using :class:`~medkit.training.Trainer`, and then
        used in a new `HFEntityMatcher` operation.
        """
        return HFEntityMatcherTrainable(
            model_name_or_path=model_name_or_path,
            labels=labels,
            tagging_scheme=tagging_scheme,
            tokenizer_max_length=tokenizer_max_length,
            tag_subtokens=tag_subtokens,
            hf_auth_token=hf_auth_token,
            device=device,
        )
