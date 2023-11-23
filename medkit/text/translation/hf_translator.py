"""
This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[hf-translator]`.
"""

from __future__ import annotations

__all__ = ["HFTranslator"]

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import transformers
from transformers import TranslationPipeline, BertModel, BertTokenizerFast

from medkit.core import Operation
from medkit.core.text import Segment, ModifiedSpan, span_utils
import medkit.core.utils


class HFTranslator(Operation):
    """Translator based on HuggingFace transformers model

    Any translation model from the HuggingFace hub can be used.

    For segment given in input, a translated segment will be returned.
    The spans of the translated segment are "aligned" to the original segment.
    An alignment model is used to find matches between translated words and
    original words, and for each of these matches a :class:`~medkit.core.text.ModifiedSpan`
    is created, referencing the original span in the original text.

    Segment given in input should not contain more than one sentence, because only the 1st
    sentence will be translated and the others will be discarded (this might vary with the model).
    The formatting will not be preserved. Note that the translation and alignment models have a
    maximum token length (typically 512) so there is a hard limit on the length of each segment anyway.
    """

    _DEFAULT_LABEL = "translation"
    _DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-fr-en"
    _DEFAULT_ALIGNMENT_MODEL = "bert-base-multilingual-cased"

    def __init__(
        self,
        output_label: str = _DEFAULT_LABEL,
        translation_model: Union[str, Path] = _DEFAULT_TRANSLATION_MODEL,
        alignment_model: Union[str, Path] = _DEFAULT_ALIGNMENT_MODEL,
        alignment_layer: int = 8,
        alignment_threshold: float = 1e-3,
        device: int = -1,  # -1 corresponds to the cpu else device number
        batch_size: int = 1,
        hf_auth_token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        uid: str = None,
    ):
        """
        Parameters
        ----------
        output_label:
            Label of the translated segments
        translation_model:
            Name (on the HuggingFace models hub) or path of the translation model. Must be a model compatible
            with the `TranslationPipeline` transformers class.
        alignment_model:
            Name (on the HuggingFace models hub) or path of the alignment model. Must be a multilingual BERT model
            compatible with the `BertModel` transformers class.
        alignment_layer:
            Index of the layer in the alignment model that contains the token embeddings
            (the original and translated embedding will be. compared)
        alignment_threshold:
            Threshold value used to decide if embeddings are similar enough to be aligned
        device:
            Device to use for transformers models. Follows the HuggingFace convention
            (-1 for "cpu" and device number for gpu, for instance 0 for "cuda:0")
        batch_size:
            Number of segments in batches processed by translation and alignment models
        hf_auth_token:
            HuggingFace Authentication token (to access private models on the
            hub)
        cache_dir:
            Directory where to store downloaded models. If not set, the default
            HuggingFace cache dir is used.
        uid:
            Identifier of the translator
        """

        # Pass all arguments to super (remove self and confidential hf_auth_token)
        init_args = locals()
        init_args.pop("self")
        init_args.pop("hf_auth_token")
        super().__init__(**init_args)

        self.output_label = output_label
        self.translation_model = translation_model
        self.alignment_model = alignment_model
        self.alignment_layer = alignment_layer
        self.alignment_threshold = alignment_threshold
        self.device = device
        self.batch_size = batch_size

        if isinstance(self.translation_model, str):
            task = transformers.pipelines.get_task(
                translation_model, token=hf_auth_token
            )
            if not task.startswith("translation"):
                raise ValueError(
                    f"Model {self.translation_model} is not associated to a translation"
                    " task and cannot be use with HFTranslator"
                )

        self._translation_pipeline = transformers.pipeline(
            task=task,
            model=self.translation_model,
            pipeline_class=TranslationPipeline,
            device=self.device,
            batch_size=self.batch_size,
            token=hf_auth_token,
            model_kwargs={"cache_dir": cache_dir},
        )
        self._aligner = _Aligner(
            model=self.alignment_model,
            layer_index=self.alignment_layer,
            threshold=self.alignment_threshold,
            device=self.device,
            batch_size=self.batch_size,
            hf_auth_token=hf_auth_token,
            cache_dir=cache_dir,
        )

    def run(self, segments: List[Segment]) -> List[Segment]:
        """
        Translate short segments (can't contain multiple sentences)

        Parameters
        ----------
        segments:
            List of segments to translate

        Returns
        -------
        List[~medkit.core.text.Segment]:
            Translated segments (with spans referring to words in original text, for translated
            words that have been aligned to original words)
        """
        return [s for s in self._translate_segments(segments)]

    def _translate_segments(self, segments: List[Segment]) -> Iterator[Segment]:
        original_texts = [s.text for s in segments]
        translated_texts = [
            d["translation_text"] for d in self._translation_pipeline(original_texts)
        ]

        # compute words alignments
        alignments = self._aligner.align(translated_texts, original_texts)

        for segment, translated_text, alignment in zip(
            segments, translated_texts, alignments
        ):
            translated_spans = self._get_translated_spans(
                alignment, translated_text, segment.text, segment.spans
            )

            translated_segment = Segment(
                label=self.output_label,
                spans=translated_spans,
                text=translated_text,
            )

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    translated_segment, self.description, source_data_items=[segment]
                )

            yield translated_segment

    def _get_translated_spans(
        self, alignment, translated_text, original_text, original_spans
    ):
        """Compute spans for translated segments, making translated words reference words
        in original text through ModifiedSpans when possible"""

        # build translated spans, which will contains:
        # - ModifiedSpans with no replacement_spans, for non-aligned parts of translated text
        #   (ie gaps between aligned words, plus head and tail)
        # - ModifiedSpans with replacement_spans pointing to original word(s) for aligned words
        #   in translated text
        # - plain Spans pointing to original word(s) for aligned words, when the translated word
        #   is identical to the original word
        translated_spans = []
        current_char = 0
        for translated_range, original_ranges in alignment.items():
            translated_start, translated_end = translated_range
            translated_sub_text = translated_text[translated_start:translated_end]

            # handle gaps between aligned sub texts
            if current_char < translated_start:
                translated_spans.append(
                    ModifiedSpan(translated_start - current_char, replaced_spans=[])
                )

            # extract spans corresponding to sub text in original text
            original_sub_text, original_sub_text_spans = span_utils.extract(
                original_text, original_spans, original_ranges
            )
            if translated_sub_text == original_sub_text:
                # if translation sub text is identical to original,
                # we can use the original spans
                translated_spans += original_sub_text_spans
            else:
                # otherwise create modified span pointing to original spans
                length = translated_end - translated_start
                translated_spans.append(
                    ModifiedSpan(length, replaced_spans=original_sub_text_spans)
                )

            current_char = translated_end

        # handle trail
        if current_char < len(translated_text):
            translated_spans.append(
                ModifiedSpan(len(translated_text) - current_char, replaced_spans=[])
            )

        assert sum(s.length for s in translated_spans) == len(translated_text)
        return translated_spans


"""Alignment data structure
Each entry in key is the character range of a word in the source text,
the value being a list of one or more character ranges of corresponding words in the target text.
Ex:
>>> source_text = "Mon prénom est Lucas"
>>> target_text = "My first name is Lucas"
>>> alignment = {
    (0, 3): [(0, 2)],  # Mon => My
    (4, 10): [(3, 8), (9, 13)], # prénom => first, name
    (11, 14): [(14, 16)],  # est => is
    (15, 20): [(17, 22)], # Lucas = Lucas
}
"""
_AlignmentDict = Dict[Tuple[int, int], List[Tuple[int, int]]]


class _Aligner:
    def __init__(
        self,
        model: Union[str, Path] = "bert-base-multilingual-cased",
        layer_index: int = 8,
        threshold: float = 1e-3,
        device: int = -1,
        batch_size: int = 1,
        hf_auth_token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self._device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self._batch_size = batch_size
        self._model = BertModel.from_pretrained(
            model,
            output_hidden_states=True,
            token=hf_auth_token,
            cache_dir=cache_dir,
        ).to(self._device)
        self._layer_index = layer_index
        self._threshold: float = threshold
        self._tokenizer = BertTokenizerFast.from_pretrained(model, token=hf_auth_token)

    def align(
        self, source_texts: List[str], target_texts: List[str]
    ) -> List[_AlignmentDict]:
        """Compute word alignments between two lists of texts in different languages.

        Parameters
        ----------
        source_texts:
            The texts to align from (typically the translated texts)
        target_texts:
            The texts to align to (typically the original texts)

        Returns
        -------
        List[_AlignmentDict]:
            List of alignments dicts between characters ranges (cf description of _AlignmentDict)
        """
        assert len(source_texts) == len(
            target_texts
        ), "Must have same number of source and target texts"

        aligments = []
        source_text_batches_iter = medkit.core.utils.batch_list(
            source_texts, self._batch_size
        )
        target_text_batches_iter = medkit.core.utils.batch_list(
            target_texts, self._batch_size
        )
        for source_text_batch, target_text_batch in zip(
            source_text_batches_iter, target_text_batches_iter
        ):
            aligments += self._align_batch(source_text_batch, target_text_batch)
        return aligments

    def _align_batch(self, source_texts, target_texts):
        # preprocess
        source_encoding = self._encode_text(source_texts)
        target_encoding = self._encode_text(target_texts)

        # extract source and target embeddings for full batch
        self._model.eval()
        with torch.no_grad():
            batch_in_source = source_encoding["input_ids"].to(self._device)
            batch_out_source = self._model(batch_in_source)
            batch_out_source = batch_out_source[2][self._layer_index]
            batch_in_target = target_encoding["input_ids"].to(self._device)
            batch_out_target = self._model(batch_in_target)
            batch_out_target = batch_out_target[2][self._layer_index]

        # compute alignment for each pair of texts in batch
        word_alignments = []
        for batch_index in range(len(source_texts)):
            # align tokens by computing similarity between embeddings forward and backwards
            out_source = batch_out_source[batch_index]
            out_target = batch_out_target[batch_index]
            dot_prod = torch.matmul(out_source, out_target.transpose(-1, -2))
            softmax_source_target = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_target_source = torch.nn.Softmax(dim=-2)(dot_prod)
            # flag as aligned where similarities are greater than threshold
            softmax_inter = (softmax_source_target > self._threshold) * (
                softmax_target_source > self._threshold
            )
            token_alignment = torch.nonzero(softmax_inter, as_tuple=False)

            # align word spans (build word alignments from token alignments, and take word spans)
            word_alignment = self._token_alignment_to_word_alignment(
                token_alignment, source_encoding, target_encoding, batch_index
            )
            word_alignments.append(word_alignment)

        return word_alignments

    def _encode_text(self, text):
        """Return a BatchEncoder instance
        (useful for converting token back to words and CharSpans,
        but requires a TokenizerFast)"""
        encodings = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._tokenizer.model_max_length,
        )
        return encodings

    def _token_alignment_to_word_alignment(
        self, token_alignment, source_encoding, target_encoding, batch_index
    ) -> _AlignmentDict:
        """Convert BERT token alignments computed from the model to word alignments,
        (using characters ranges of aligned words)"""
        source_word_ids = source_encoding.word_ids(batch_index)
        target_word_ids = target_encoding.word_ids(batch_index)

        # align word spans (build word alignments from token alignments, and take word spans)
        word_alignment = defaultdict(list)
        for source_token, target_token in token_alignment:
            source_word = source_word_ids[source_token]
            target_word = target_word_ids[target_token]
            if source_word is None or target_word is None:
                continue

            source_range = tuple(
                source_encoding.word_to_chars(batch_index, source_word)
            )
            target_range = tuple(
                target_encoding.word_to_chars(batch_index, target_word)
            )
            if target_range not in word_alignment[source_range]:
                word_alignment[source_range].append(target_range)

        # sort target ranges (token_alignment is sorted on 1st column (source)
        # but not necessarily on 2d column (target), ie. it is not monotonic)
        for target_ranges in word_alignment.values():
            target_ranges.sort()

        return word_alignment
