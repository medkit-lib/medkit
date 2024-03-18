"""Tools for accessing examples of mtsamples files.

Refer to the `mtsamplesFR`_ repository for more information.

This repository contains:

* the original dataset from Kaggle (data/mtsamples.csv);

* a French translation for the dataset (data/mtsamples_translation.json).

Both of which are made available under the CC0-1.0 license.

.. _mtsamplesFR: https://github.com/medkit-lib/mtsamplesFR
"""

from __future__ import annotations

__all__ = ["load_mtsamples", "convert_mtsamples_to_medkit"]

import csv
import json
import urllib.request
from pathlib import Path

from medkit.core.text import TextDocument
from medkit.io.medkit_json import save_text_documents

_REPO_URL: str = "https://raw.githubusercontent.com/medkit-lib/mtsamplesFR/master/data/"
_MTSAMPLES_FILE: str = "mtsamples.csv"
_MTSAMPLES_TRANSLATED_FILE: str = "mtsamples_translated.json"


def load_mtsamples(
    cache_dir: Path | str = ".cache",
    translated: bool = True,
    nb_max: int | None = None,
) -> list[TextDocument]:
    """Load mtsamples data as medkit text documents.

    Parameters
    ----------
    cache_dir : str or Path, default=".cache"
        Directory where to store mtsamples file. Default: .cache
    translated : bool, default=True
        If True (default), `mtsamples_translated.json` file is used (FR).
        If False, `mtsamples.csv` is used (EN)
    nb_max : int, optional
        Maximum number of documents to load

    Returns
    -------
    list of TextDocument
        The medkit text documents corresponding to mtsamples data
    """
    if translated:
        mtsamples_url = _REPO_URL + _MTSAMPLES_TRANSLATED_FILE
        cache_file = Path(cache_dir) / Path(_MTSAMPLES_TRANSLATED_FILE)
    else:
        mtsamples_url = _REPO_URL + _MTSAMPLES_FILE
        cache_file = Path(cache_dir) / Path(_MTSAMPLES_FILE)

    if not cache_file.exists():
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        urllib.request.urlretrieve(mtsamples_url, cache_file)  # noqa: S310

    with cache_file.open() as fp:
        mtsamples = json.load(fp) if translated else list(csv.DictReader(fp))

        if nb_max is not None:
            mtsamples = mtsamples[:nb_max]

        return [
            TextDocument(
                text=(sample["transcription_translated"] if translated else sample["transcription"]),
                metadata={
                    "id": sample["id"] if translated else sample[""],
                    "description": sample["description"],
                    "medical_specialty": sample["medical_specialty"],
                    "sample_name": sample["sample_name"],
                    "keywords": sample["keywords"],
                },
            )
            for sample in mtsamples
        ]


def convert_mtsamples_to_medkit(
    output_file: Path | str,
    encoding: str | None = "utf-8",
    cache_dir: Path | str = ".cache",
    translated: bool = True,
):
    """Save mtsamples data as a medkit text file.

    Parameters
    ----------
    output_file : str or Path
        Path to the medkit jsonl file to generate
    encoding : str, default="utf-8"
        Encoding of the medkit file to generate
    cache_dir : str or Path, default=".cache"
        Directory where mtsamples file is cached. Default: .cache
    translated : bool, default=True
        If True (default), `mtsamples_translated.json` file is used (FR).
        If False, `mtsamples.csv` is used (EN)

    """
    docs = load_mtsamples(cache_dir, translated)
    save_text_documents(docs=docs, output_file=output_file, encoding=encoding)
