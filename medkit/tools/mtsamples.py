"""
This module aims to provide facilities for accessing some examples of mtsamples files
available on this repository: https://github.com/neurazlab/mtsamplesFR

Refer to the repository for more information.

This repository contains:

* **a version of mtsamples.csv**
    Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
    license: CC0: Public Domain

* **a mtsamples_translation.json file which is a translation to french**

Date: 08/04/2022
"""

__all__ = ["load_mtsamples", "convert_mtsamples_to_medkit"]

import csv
import json
import urllib.request
from pathlib import Path
from typing import List, Optional, Union
from medkit.core.text import TextDocument
from medkit.io.medkit_json import save_text_documents


_REPO_URL: str = "https://raw.githubusercontent.com/aneuraz/mtsamplesFR/master/data/"
_MTSAMPLES_FILE: str = "mtsamples.csv"
_MTSAMPLES_TRANSLATED_FILE: str = "mtsamples_translated.json"


def load_mtsamples(
    cache_dir: Union[Path, str] = ".cache",
    translated: bool = True,
    nb_max: Optional[int] = None,
) -> List[TextDocument]:
    """
    Function loading mtsamples data into medkit text documents

    Parameters
    ----------
    cache_dir
        Directory where to store mtsamples file. Default: .cache
    translated
        If True (default), `mtsamples_translated.json` file is used (FR).
        If False, `mtsamples.csv` is used (EN)
    nb_max
        Maximum number of documents to load

    Returns
    -------
    List[TextDocument]
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
        urllib.request.urlretrieve(
            mtsamples_url,
            cache_file,
        )

    with open(cache_file) as f:
        if translated:
            mtsamples = json.load(f)
        else:
            mtsamples = csv.DictReader(f)

        if nb_max is not None:
            mtsamples = mtsamples[:nb_max]

        return [
            TextDocument(
                text=(
                    sample["transcription_translated"]
                    if translated
                    else sample["transcription"]
                ),
                metadata=dict(
                    id=sample["id"] if translated else sample[""],
                    description=sample["description"],
                    medical_specialty=sample["medical_specialty"],
                    sample_name=sample["sample_name"],
                    keywords=sample["keywords"],
                ),
            )
            for sample in mtsamples
        ]


def convert_mtsamples_to_medkit(
    output_file: Union[Path, str],
    encoding: Optional[str] = "utf-8",
    cache_dir: Union[Path, str] = ".cache",
    translated: bool = True,
):
    """
    Convert mtsamples data into  a medkit file

    Parameters
    ----------
    output_file
        Path to the medkit jsonl file to generate
    encoding
        Encoding of the medkit file to generate
    cache_dir
        Directory where mtsamples file is cached. Default: .cache
    translated
        If True (default), `mtsamples_translated.json` file is used (FR).
        If False, `mtsamples.csv` is used (EN)

    """
    docs = load_mtsamples(cache_dir, translated)
    save_text_documents(docs=docs, output_file=output_file, encoding=encoding)
