import pytest

from medkit.io.medkit_json import load_text_documents
from medkit.tools.mtsamples import convert_mtsamples_to_medkit, load_mtsamples


@pytest.mark.parametrize(
    ("translated", "expected_header"),
    [
        (True, "SUBJECTIF :, Cette femme blanche de 23 ans"),
        (False, "SUBJECTIVE:,  This 23-year-old"),
    ],
    ids=[
        "french_mtsamples",
        "english_mtsamples",
    ],
)
def test_convert_mtsamples_to_medkit(tmpdir, translated, expected_header):
    medkit_file = tmpdir / "medkit.jsonl"
    convert_mtsamples_to_medkit(output_file=medkit_file, translated=translated)
    doc = load_mtsamples(nb_max=1, translated=translated)[0]

    docs_from_medkit = load_text_documents(input_file=medkit_file)
    assert doc.text.startswith(expected_header)
    assert doc.text == docs_from_medkit.__next__().text
