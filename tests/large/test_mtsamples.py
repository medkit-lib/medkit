import json
import logging
from pathlib import Path

from medkit.core.doc_pipeline import DocPipeline
from medkit.core.pipeline import Pipeline, PipelineStep
from medkit.core.text import TextDocument
from medkit.text.context import NegationDetector
from medkit.text.ner import RegexpMatcher
from medkit.text.preprocessing import RegexpReplacer
from medkit.text.segmentation import SentenceTokenizer

_PATH_TO_MTSAMPLES = Path(__file__).parent / ".." / "data" / "mtsamples"


def _get_medkit_docs():
    path = _PATH_TO_MTSAMPLES / "mtsamples_translated.json"
    if not path.exists():
        msg = (
            "For running this test, you need to have mtsamples_translated.json file in"
            " `tests/data/mtsamples` folder.\nThe file is not provided with medkit"
            " library. Please contact us to get this file."
        )
        raise FileNotFoundError(msg)
    with Path(path).open() as fp:
        dataset = json.load(fp)

    docs = []
    for data in dataset:
        metadata = {}
        text = ""
        for key, value in data.items():
            if key == "transcription_translated":
                text = value
            else:
                metadata[key] = value
        docs.append(TextDocument(text=text, metadata=metadata))

    return docs


def test_mt_samples_without_pipeline(caplog):
    docs = _get_medkit_docs()
    assert len(docs) == 4999

    # init and configure operations
    rules = [
        (r"[nN]\s*°", "numéro"),
        (r"(?<=[0-9]\s)°", " degrés"),
        (r"(?<=[0-9])°", " degrés"),
        ("\u00c6", "AE"),  # ascii
        ("\u00e6", "ae"),  # ascii
        ("\u0152", "OE"),  # ascii
        ("\u0153", "oe"),  # ascii
        (r"«|»", '"'),
        ("®|©", ""),
        ("½", "1/2"),  # ascii
        ("…", "..."),  # ascii
        ("¼", "1/4"),  # ascii
    ]
    regexp_replacer = RegexpReplacer(output_label="norm_text", rules=rules)
    sentence_tokenizer = SentenceTokenizer()
    negation_detector = NegationDetector(output_label="negation")
    regexp_matcher = RegexpMatcher(attrs_to_copy=["negation"])

    # annotate each doc
    nb_tot_anns = 0
    for doc in docs:
        anns = [doc.raw_segment]
        anns = regexp_replacer.run(anns)
        anns = sentence_tokenizer.run(anns)
        with caplog.at_level(logging.WARNING, logger="medkit.text.context.negation_detector"):
            negation_detector.run(anns)
            assert len(caplog.messages) == 0
        with caplog.at_level(logging.WARNING, logger="medkit.text.context.regexp_matcher"):
            anns = regexp_matcher.run(anns)
            assert len(caplog.messages) == 0

        for ann in anns:
            doc.anns.add(ann)

        nb_tot_anns += len(doc.anns)

    assert nb_tot_anns == 13631


def test_mt_samples_with_doc_pipeline():
    docs = _get_medkit_docs()
    assert len(docs) == 4999

    # init and configure operations
    rules = [
        (r"[nN]\s*°", "numéro"),
        (r"(?<=[0-9]\s)°", " degrés"),
        (r"(?<=[0-9])°", " degrés"),
        ("\u00c6", "AE"),  # ascii
        ("\u00e6", "ae"),  # ascii
        ("\u0152", "OE"),  # ascii
        ("\u0153", "oe"),  # ascii
        (r"«|»", '"'),
        ("®|©", ""),
        ("½", "1/2"),  # ascii
        ("…", "..."),  # ascii
        ("¼", "1/4"),  # ascii
    ]

    char_replacer = PipelineStep(
        operation=RegexpReplacer(output_label="norm_text", rules=rules),
        input_keys=["full_text"],
        output_keys=["norm_text"],
    )
    sentence_tokenizer = PipelineStep(
        operation=SentenceTokenizer(),
        input_keys=["norm_text"],
        output_keys=["sentences"],
    )
    negation_detector = PipelineStep(
        operation=NegationDetector(output_label="negation"),
        input_keys=["sentences"],
        output_keys=[],
    )

    regexp_matcher = PipelineStep(
        operation=RegexpMatcher(attrs_to_copy=["negation"]),
        input_keys=["sentences"],
        output_keys=["entities"],
    )

    pipeline = Pipeline(
        steps=[char_replacer, sentence_tokenizer, negation_detector, regexp_matcher],
        input_keys=char_replacer.input_keys,
        output_keys=regexp_matcher.output_keys,
    )

    doc_pipeline = DocPipeline(
        pipeline=pipeline,
        labels_by_input_key={"full_text": [TextDocument.RAW_LABEL]},
    )
    # annotate each doc
    nb_tot_anns = 0

    doc_pipeline.run(docs)

    for doc in docs:
        nb_tot_anns += len(doc.anns)

    assert nb_tot_anns == 13631
