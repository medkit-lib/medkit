"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[hf-entity-matcher]`.
"""
from __future__ import annotations

__all__ = ["HFEntityMatcherTrainable"]

import logging
from typing import TYPE_CHECKING, Any

import torch
import transformers
from typing_extensions import Literal

from medkit.text.ner import hf_tokenization_utils
from medkit.tools import hf_utils
from medkit.training.utils import BatchData

if TYPE_CHECKING:
    from pathlib import Path

    from medkit.core.text import Entity, TextDocument

logger = logging.getLogger(__name__)


class HFEntityMatcherTrainable:
    """Trainable entity matcher based on HuggingFace transformers model
    Any token classification model from the HuggingFace hub can be used
    (for instance "samrawal/bert-base-uncased_clinical-ner").
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        labels: list[str],
        tagging_scheme: Literal["bilou", "iob2"],
        tag_subtokens: bool = False,
        tokenizer_max_length: int | None = None,
        hf_auth_token: str | None = None,
        device: int = -1,
    ):
        """Parameters
        ----------
        model_name_or_path : str or Path
            Name (on the HuggingFace models hub) or path of the NER model. Must be a model compatible
            with the `TokenClassification` transformers class.
        labels : list of str
            List of labels to detect
        tagging_scheme : {"bilou", "iob2"}
            Tagging scheme to use in the segment-entities preprocessing and label mapping definition.
        tag_subtokens : bool, default=False
            Whether tag subtokens in a word. PreTrained models require a tokenization step.
            If any word of the segment is not in the vocabulary of the tokenizer used by the PreTrained model,
            the word is split into subtokens.
            It is recommended to only tag the first subtoken of a word. However, it is possible to tag all subtokens
            by setting this value to `True`. It could influence the time and results of fine-tunning.
        tokenizer_max_length : int, optional
            Optional max length for the tokenizer, by default the `model_max_length` will be used.
        hf_auth_token : str, optional
            HuggingFace Authentication token (to access private models on the
            hub)
        device : int, default=-1
            Device to use for the transformer model. Follows the HuggingFace convention
            (-1 for "cpu" and device number for gpu, for instance 0 for "cuda:0").
        """
        valid_model = hf_utils.check_model_for_task_hf(
            model_name_or_path, "token-classification", hf_auth_token=hf_auth_token
        )
        if not valid_model:
            msg = (
                f"Model {model_name_or_path} is not associated to a"
                " token-classification/ner task and cannot be used with"
                " HFEntityMatcher"
            )
            raise ValueError(msg)

        self.model_name_or_path = model_name_or_path
        self.tagging_scheme = tagging_scheme
        self.tag_subtokens = tag_subtokens
        self.tokenizer_max_length = tokenizer_max_length
        self.model_config = self._get_valid_model_config(labels)

        # update labels mapping using the configuration
        self.label_to_id = self.model_config.label2id
        self.id_to_label = self.model_config.id2label

        # load tokenizer and model using the model path
        self.load(self.model_name_or_path, hf_auth_token=hf_auth_token)
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self._model.to(self.device)

        # init data collator in charge of padding
        self._data_collator = transformers.DataCollatorForTokenClassification(tokenizer=self._tokenizer)

    def configure_optimizer(self, lr: float) -> torch.optim.Optimizer:
        # todo: group_params optimizer_parameters = [{}]
        optimizer_parameters = self._model.parameters()
        return torch.optim.AdamW(optimizer_parameters, lr=lr)

    def preprocess(self, data_item: TextDocument) -> dict[str, Any]:
        # tokenize each and compute corresponding labels for each tokens
        # (no padding for now, this will be done at the collate stage)

        text_encoding = self._encode_text(data_item.text)
        entities: list[Entity] = data_item.anns.entities

        tags = hf_tokenization_utils.transform_entities_to_tags(
            entities=entities,
            text_encoding=text_encoding,
            tagging_scheme=self.tagging_scheme,
        )
        tags_ids = hf_tokenization_utils.align_and_map_tokens_with_tags(
            text_encoding=text_encoding,
            tags=tags,
            tag_to_id=self.label_to_id,
            map_sub_tokens=self.tag_subtokens,
        )

        model_input = {}
        model_input["input_ids"] = text_encoding.ids
        model_input["labels"] = tags_ids

        return model_input

    def _encode_text(self, text):
        """Return a EncodingFast instance"""
        text_tokenized = self._tokenizer(
            text,
            padding="do_not_pad",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_special_tokens_mask=True,
        )
        return text_tokenized.encodings[0]

    def collate(self, batch: list[dict[str, Any]]) -> BatchData:
        # rely on transformer's collator to handle padding
        batch = self._data_collator(features=batch)
        # wrap results in our own data structure
        return BatchData(batch)

    def forward(
        self,
        input_batch: BatchData,
        return_loss: bool,
        eval_mode: bool,
    ) -> tuple[BatchData, torch.Tensor | None]:
        if eval_mode:
            self._model.eval()
        else:
            self._model.train()

        model_output = self._model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            labels=input_batch["labels"],
        )
        loss = model_output["loss"] if return_loss else None
        return BatchData(logits=model_output["logits"]), loss

    def save(self, path: str | Path):
        state_dict = self._model.state_dict()
        self._model.save_pretrained(path, state_dict=state_dict)
        self._tokenizer.save_pretrained(path)

    def load(self, path: str | Path, hf_auth_token: str | None = None):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            model_max_length=self.tokenizer_max_length,
            token=hf_auth_token,
        )

        if not isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            msg = (
                "This operation only works with model that have a fast tokenizer. Check"
                " the hugging face documentation to find the required tokenizer"
            )
            raise TypeError(msg)

        # we intentionally do not pad at the encoding stage
        # so we disable this warning
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        model = transformers.AutoModelForTokenClassification.from_pretrained(
            path,
            config=self.model_config,
            ignore_mismatched_sizes=True,
            token=hf_auth_token,
        )

        self._tokenizer = tokenizer
        self._model = model

    def _get_valid_model_config(self, labels: list[str], hf_auth_token: str | None = None):
        """Return a config file with the correct mapping of labels"""
        # get possible tags from labels list
        label_to_id = hf_tokenization_utils.convert_labels_to_tags(labels=labels, tagging_scheme=self.tagging_scheme)
        nb_labels = len(label_to_id)

        # load configuration with the correct number of NER labels
        config = transformers.AutoConfig.from_pretrained(
            self.model_name_or_path, num_labels=nb_labels, token=hf_auth_token
        )

        # If the model has the same labels, we kept the original mapping
        # Easier finetunning
        if sorted(config.label2id.keys()) != sorted(label_to_id.keys()):
            logger.warning(
                """The operation model seems to have different labels.
                PreTrained with labels: %s, new labels %s.
                Ignoring the model labels as result.""",
                sorted(config.label2id.keys()),
                sorted(label_to_id.keys()),
            )
            config.label2id = dict(label_to_id.items())
            config.id2label = {idx: label for label, idx in label_to_id.items()}

        return config
