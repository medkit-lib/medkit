"""This package needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[metrics-ner]`.
"""
from __future__ import annotations

__all__ = ["SeqEvalEvaluator", "SeqEvalMetricsComputer"]

from typing import TYPE_CHECKING, Any

from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import BILOU, IOB2
from typing_extensions import Literal

from medkit.core.text import Entity, TextDocument, span_utils
from medkit.text.ner import hf_tokenization_utils

if TYPE_CHECKING:
    from medkit.training.utils import BatchData


def _compute_seqeval_from_dict(
    y_true_all: list[list[str]],
    y_pred_all: list[list[str]],
    tagging_scheme: Literal["bilou", "iob2"],
    return_metrics_by_label: bool,
    average: Literal["macro", "weighted"],
) -> dict[str, float | int]:
    """Compute seqeval metrics using preprocessed data"""
    # internal configuration for seqeval
    # 'bilou' only works with 'strict' mode
    scheme = BILOU if tagging_scheme == "bilou" else IOB2
    mode = "strict" if tagging_scheme == "bilou" else None

    # returns precision, recall, F1 score for each class.
    report = classification_report(
        y_true=y_true_all,
        y_pred=y_pred_all,
        scheme=scheme,
        output_dict=True,
        zero_division=0,
        mode=mode,
    )
    # add average metrics
    scores = {f"{average}_{key}": value for key, value in report[f"{average} avg"].items()}
    scores["support"] = scores.pop(f"{average}_support")
    scores["accuracy"] = accuracy_score(y_true=y_true_all, y_pred=y_pred_all)

    if return_metrics_by_label:
        for value_key in report:
            if value_key.endswith("avg"):
                continue
            for metric_key, metric_value in report[value_key].items():
                scores[f"{value_key}_{metric_key}"] = metric_value

    return scores


class SeqEvalEvaluator:
    """Evaluator to compute the performance of labeling tasks such as
    named entity recognition. This evaluator compares TextDocuments of reference
    with its predicted annotations and returns a dictionary of metrics.

    The evaluator converts the set of entities and documents to tags before compute the metric.
    It supports two schemes, IOB2 (a BIO scheme) and BILOU. The IOB2 scheme tags the Beginning,
    the Inside and the Outside text of a entity. The BILOU scheme tags the Beginning,
    the Inside and the Last tokens of multi-token entity as well as Unit-length entity.

    For more information about IOB schemes, refer to the `Wikipedia page <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>`_

    .. hint::
        If **tokenizer** is not defined, the evaluator tokenizes the text by character.
        This may generate a lot of tokens with large documents and may affect execution time.
        You can use a fast tokenizer from HuggingFace, i.e. : bert tokenizer

        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    """

    def __init__(
        self,
        tagging_scheme: Literal["bilou", "iob2"] = "bilou",
        return_metrics_by_label: bool = True,
        average: Literal["macro", "weighted"] = "macro",
        tokenizer: Any | None = None,
        labels_remapping: dict[str, str] | None = None,
    ):
        """
        Parameters
        ----------
        tagging_scheme : str, default="bilou"
            Scheme for tagging the tokens, it can be `bilou` or `iob2`
        return_metrics_by_label : bool, default=True
            If `True`, return the metrics by label in the output dictionary.
            If `False`, only global metrics are returned
        average : str, default="macro"
            Type of average to be performed in metrics.
            - `macro`, unweighted mean (default)
            - `weighted`, weighted average by support (number of true instances by label)
        tokenizer : Any, optional
            Optional Fast Tokenizer to convert text into tokens.
            If not provided, the text is tokenized by character.
        labels_remapping : dict of str to str, optional
            Optional remapping of labels, useful when there is a mismatch
            between the predicted labels and the reference labels to evaluate
            against. If a label (of a reference of predicted entity) is found in
            this dict, the corresponding value will be used as label instead.
        """
        self.tokenizer = tokenizer
        self.tagging_scheme = tagging_scheme
        self.return_metrics_by_label = return_metrics_by_label
        self.average = average
        self.labels_remapping = labels_remapping

    def compute(self, documents: list[TextDocument], predicted_entities: list[list[Entity]]) -> dict[str, float]:
        """Compute metrics of entity matching giving predictions.

        Parameters
        ----------
        documents : list of TextDocuments
            Text documents containing entities of reference
        predicted_entities : list of list of Entity
            List of predicted entities by document

        Returns
        -------
        dict of str to float
            A dictionary with average and per type metrics if required. The metrics included are:
            accuracy, precision, recall and F1 score.
        """
        true_tags_all, pred_tags_all = [], []

        for document, pred_entities in zip(documents, predicted_entities):
            text = document.text
            true_entities = document.anns.entities

            true_tags_all.append(self._tag_text_with_entities(text=text, entities=true_entities))
            pred_tags_all.append(self._tag_text_with_entities(text=text, entities=pred_entities))
        return _compute_seqeval_from_dict(
            y_true_all=true_tags_all,
            y_pred_all=pred_tags_all,
            tagging_scheme=self.tagging_scheme,
            return_metrics_by_label=self.return_metrics_by_label,
            average=self.average,
        )

    def _tag_text_with_entities(self, text: str, entities: list[Entity]):
        if self.tokenizer is not None:
            # tags tokenized text, creates one tag per token
            text_encoding = self.tokenizer(text).encodings[0]
            return hf_tokenization_utils.transform_entities_to_tags(
                text_encoding=text_encoding,
                entities=entities,
                tagging_scheme=self.tagging_scheme,
            )

        # tags untokenized text, create one tag per character
        tags = ["O"] * len(text)
        for ent in entities:
            label = ent.label
            if self.labels_remapping:
                label = self.labels_remapping.get(label, label)
            ent_spans = span_utils.normalize_spans(ent.spans)
            # skip if all spans were ModifiedSpans and we are
            # not able to refer back to text
            if not ent_spans:
                continue

            start_char = ent_spans[0].start
            end_char = ent_spans[-1].end
            chars_entity = list(range(start_char, end_char))

            if not chars_entity:
                continue

            entity_tags = hf_tokenization_utils.create_entity_tags(
                nb_tags=len(chars_entity),
                label=label,
                tagging_scheme=self.tagging_scheme,
            )
            for token_idx, tag in zip(chars_entity, entity_tags):
                tags[token_idx] = tag

        return tags


class SeqEvalMetricsComputer:
    """An implementation of :class:`~medkit.training.MetricsComputer` using seqeval
    to compute metrics in the training of named-entity recognition components.

    The metrics computer can be used with a :class:`~medkit.training.Trainer`
    """

    def __init__(
        self,
        id_to_label: dict[int, str],
        tagging_scheme: Literal["bilou", "iob2"] = "bilou",
        return_metrics_by_label: bool = True,
        average: Literal["macro", "weighted"] = "macro",
    ):
        """
        Parameters
        ----------
        id_to_label : dict of int to str
            Mapping integer value to label, it should be the same used in preprocess
        tagging_scheme : str, default="bilou"
            Scheme used for tagging the tokens, it can be `bilou` or `iob2`
        return_metrics_by_label : bool, default=True
            If `True`, return the metrics by label in the output dictionary.
            If `False`, only return average metrics
        average : str, default="macro"
            Type of average to be performed in metrics.
            - `macro`, unweighted mean (default)
            - `weighted`, weighted average by support (number of true instances by attr value)
        """
        self.id_to_label = id_to_label
        self.tagging_scheme = tagging_scheme
        self.return_metrics_by_label = return_metrics_by_label
        self.average = average

    def prepare_batch(self, model_output: BatchData, input_batch: BatchData) -> dict[str, list[list[str]]]:
        """Prepare a batch of tensors to compute the metric

        Parameters
        ----------
        model_output : BatchData
            A batch data including the `logits` predicted by the model
        input_batch : BatchData
            A batch data including the `labels` of reference

        Returns
        -------
        dict of str to list of list of str
            A dictionary with the true and predicted tags representation of a batch data
        """
        predictions_ids = model_output["logits"].argmax(dim=-1).detach().to("cpu").numpy()
        references_ids = input_batch["labels"].detach().to("cpu").numpy()
        # ignore special tokens
        mask_special_tokens = references_ids != hf_tokenization_utils.SPECIAL_TAG_ID_HF

        batch_true_tags = [
            [self.id_to_label[tag] for tag in ref[mask]] for ref, mask in zip(references_ids, mask_special_tokens)
        ]

        batch_pred_tags = [
            [self.id_to_label[tag] for tag in pred[mask]] for pred, mask in zip(predictions_ids, mask_special_tokens)
        ]

        return {"y_true": batch_true_tags, "y_pred": batch_pred_tags}

    def compute(self, all_data: dict[str, list[Any]]) -> dict[str, float]:
        """Compute metrics using the tag representation collected by batches
        during the training/evaluation loop.

        Parameters
        ----------
        all_data : dict of str to list of Any
            A dictionary with the true and predicted tags collected by batches

        Returns
        -------
        dict of str to float
            A dictionary with average and per label metrics if required. The metrics
            included are : accuracy, precision, recall and F1 score.

        """
        # extract and format data from all_data
        y_true_all = all_data.get("y_true", [])
        y_pred_all = all_data.get("y_pred", [])
        return _compute_seqeval_from_dict(
            y_pred_all=y_pred_all,
            y_true_all=y_true_all,
            tagging_scheme=self.tagging_scheme,
            return_metrics_by_label=self.return_metrics_by_label,
            average=self.average,
        )
