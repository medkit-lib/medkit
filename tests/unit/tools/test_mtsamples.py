from medkit.io.medkit_json import load_text_documents
from medkit.tools.mtsamples import convert_mtsamples_to_medkit, load_mtsamples


def test_convert_mtsamples_to_medkit(tmpdir):
    medkit_file = tmpdir / "medkit.jsonl"
    convert_mtsamples_to_medkit(output_file=medkit_file)
    doc = load_mtsamples(nb_max=1)[0]

    docs_from_medkit = load_text_documents(input_file=medkit_file)
    assert doc.text.startswith("SUBJECTIF :, Cette femme blanche de 23 ans")
    assert doc.text == docs_from_medkit.__next__().text
