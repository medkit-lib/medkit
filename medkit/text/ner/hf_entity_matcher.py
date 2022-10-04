__all__ = ["HFEntityMatcher"]

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from typing_extensions import Literal

import transformers
from transformers import TokenClassificationPipeline

from medkit.core import Attribute
from medkit.core.text import NEROperation, Segment, span_utils, Entity


class HFEntityMatcher(NEROperation):
    """
    Entity matcher based on HuggingFace transformers model

    Any token classification model from the HuggingFace hub can be used
    (for instance "samrawal/bert-base-uncased_clinical-ner").
    """

    def __init__(
        self,
        model: Union[str, Path],
        aggregation_strategy: Literal[
            "none", "simple", "first", "average", "max"
        ] = "max",
        attrs_to_copy: Optional[List[str]] = None,
        device: int = -1,
        batch_size: int = 1,
        op_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model:
            Name (on the HuggingFace models hub) or path of the NER model. Must be a model compatible
            with the `TokenClassification` transformers class.
        aggregation_strategy:
            Strategy to fuse tokens based on the model prediction, passed to `TokenClassificationPipeline`.
            Defaults to "max", cf https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy
            for details
        attrs_to_copy:
            Labels of the attributes that should be copied from the input segment
            to the created entity. Useful for propagating context attributes
            (negation, antecendent, etc).
        device:
            Device to use for the transformer model. Follows the HuggingFace convention
            (-1 for "cpu" and device number for gpu, for instance 0 for "cuda:0").
        batch_size:
            Number of segments in batches processed by the transformer model.
        op_id:
            Identifier of the matcher.
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.model = model
        self.attrs_to_copy = attrs_to_copy

        if isinstance(self.model, str):
            task = transformers.pipelines.get_task(self.model)
            if task != "token-classification":
                raise ValueError(
                    f"Model {self.model} is not associated to a"
                    " token-classification/ner task and cannot be used with"
                    " HFEntityMatcher"
                )

        self._pipeline = transformers.pipeline(
            task=task,
            model=self.model,
            aggregation_strategy=aggregation_strategy,
            pipeline_class=TokenClassificationPipeline,
            device=device,
            batch_size=batch_size,
        )

    def run(self, segments: List[Segment]) -> List[Entity]:
        """Return entities for each match in `segments`.

        Parameters
        ----------
        segments:
            List of segments into which to look for matches.

        Returns
        -------
        List[Entity]
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

    def _matches_to_entities(
        self, matches: List[Dict], segment: Segment
    ) -> Iterator[Entity]:
        for match in matches:
            text, spans = span_utils.extract(
                segment.text, segment.spans, [(match["start"], match["end"])]
            )

            entity = Entity(
                label=match["entity_group"],
                text=text,
                spans=spans,
            )

            for label in self.attrs_to_copy:
                for attr in segment.get_attrs_by_label(label):
                    entity.add_attr(attr)

            score_attr = Attribute(
                label="score",
                value=float(match["score"]),
                metadata=dict(model=self.model),
            )
            entity.add_attr(score_attr)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    entity, self.description, source_data_items=[segment]
                )
                self._prov_tracer.add_prov(
                    score_attr, self.description, source_data_items=[segment]
                )

            yield entity