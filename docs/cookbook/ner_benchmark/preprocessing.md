# Preprocessing

```{code} python
from glob import glob
from medkit.core.text import TextDocument
from medkit.io.brat import BratInputConverter
from medkit.text.postprocessing import filter_overlapping_entities, DocumentSplitter
from statistics import mean
import pandas as pd
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
from medkit.core import Pipeline, DocPipeline, PipelineStep
from pathlib import Path
from medkit.tools.e3c_corpus import load_data_annotation
from medkit.io.doccano import DoccanoInputConverter,DoccanoTask
from sklearn.model_selection import train_test_split
from medkit.tools.mtsamples import load_mtsamples
from medkit.io.medkit_json import save_text_documents
import json

sentence_tok = SentenceTokenizer(output_label="sentence", punct_chars=["."],keep_punct=True,split_on_newlines=True)

pipeline_phrase_creator = Pipeline(steps=[PipelineStep(sentence_tok, input_keys=["full_text"], output_keys=["sentences"])],
                                          input_keys=["full_text"],
                                          output_keys=["sentences"])

phrase_creator = DocPipeline(pipeline_phrase_creator)

splitter = DocumentSplitter(segment_label="sentence", attr_labels=[])

def corpus_specs(_corpus, _title, num_docs):
    doc_data = {}
    doc_data['Documents'] = num_docs
    doc_data['Sentences'] = len(_corpus)
    doc_data['MSL'] = round(mean([len(sen.text) for sen in _corpus]))
    doc_data['All'] = sum([len(doc.anns.get_entities()) for doc in _corpus])

    labels = []
    for doc in _corpus:
        for ent in doc.anns.get_entities():
            if ent.label not in doc_data:
                doc_data[ent.label] = 0
                labels.append(ent.label)
            doc_data[ent.label] += 1

    for label in labels:
        doc_data[label] = round(doc_data[label] / doc_data['All'] * 100)

    df = pd.DataFrame(doc_data, index=[_title])
    return df

def load_quaero_split(_split):
    QUAERO_DIR = Path.home() / "src/corpus/QUAERO_FrenchMed/corpus"
    converter = BratInputConverter()
    raw_docs = []

    for text_file in sorted(QUAERO_DIR.glob(f"{_split}/*/*.txt")):
        doc = TextDocument.from_file(text_file)
        ann_file = text_file.with_suffix(".ann")
        entities = converter.load_annotations(ann_file)
        entities = filter_overlapping_entities(entities)
        for ent in entities:
            doc.anns.add(ent)
        raw_docs.append(doc)

    phrase_creator.run(raw_docs)
    splitted_docs = splitter.run(raw_docs)

    return splitted_docs, corpus_specs(splitted_docs, num_docs=len(raw_docs), _title=_split)

def load_e3c_split(_split):
    data_collection = Path.home() / "src/corpus/E3C_corpus"
    dir_path = data_collection / _split
    raw_docs = list(load_data_annotation(dir_path=dir_path, keep_sentences = True))

    for doc in raw_docs:
        for ent in doc.anns.get_entities():
            ent.label = "DISO"

    phrase_creator.run(raw_docs)
    splitted_docs = splitter.run(raw_docs)

    return splitted_docs, corpus_specs(splitted_docs, num_docs=len(raw_docs), _title=_split)

def load_casm2():

    ANNOTATION_DIR = Path.home() / "src/corpus/CasM2_Files/m2annotations"
    SPLIT_SEEDS = (67, 33)
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2

    converter = DoccanoInputConverter(task=DoccanoTask.RELATION_EXTRACTION)
    raw_documents = converter.load_from_directory_zip(ANNOTATION_DIR)

    phrase_creator.run(raw_documents)

    casm2 = {}
    casm2['train'], casm2['test'] = train_test_split(raw_documents, random_state=SPLIT_SEEDS[0], test_size=TEST_SIZE)
    casm2['train'], casm2['val'] = train_test_split(casm2['train'], random_state=SPLIT_SEEDS[1], test_size=VALIDATION_SIZE)
    casm2_splitter = DocumentSplitter(segment_label="sentence", entity_labels=['treatment', 'test', 'problem'], attr_labels=[])

    remap = {'treatment':'CHEM', 'test':'PROC','problem':'DISO'}
    casm2_splitted = {}
    docs_num = {}

    for key, docs in casm2.items():
        docs_num[key] = len(docs)
        casm2_splitted[key] = casm2_splitter.run(docs)
        for doc in casm2_splitted[key]:
            for ent in doc.anns.get_entities():
                if ent.label in remap:
                    ent.label = remap[ent.label]

    specs = pd.concat([corpus_specs(casm2_splitted[key], key, docs_num[key]) for key in casm2.keys()])

    return casm2_splitted, specs

def load_quaero():
    splits = ["train", "test", "dev"]

    quaero = {}
    stats = []

    for split in splits:
        quaero[split], stat = load_quaero_split(split)
        stats.append(stat)

    specs = pd.concat(stats)
    quaero['val'] = quaero.pop('dev')

    return quaero, specs

def load_e3c():
    splits = ["layer1_test", "layer1_train", "layer2_val"]

    e3c = {}
    stats = []

    for split in splits:
        e3c[split], stat = load_e3c_split(split)
        stats.append(stat)

    specs = pd.concat(stats)

    e3c['test'] = e3c.pop('layer1_test')
    e3c['train'] = e3c.pop('layer1_train')
    e3c['val'] = e3c.pop('layer2_val')

    return e3c, specs

def load_processed_mtsamples():
    mt_samples = load_mtsamples()
    doc_num = len(mt_samples)
    phrase_creator.run(mt_samples)
    mt_splitted = splitter.run(mt_samples)
    specs = corpus_specs(mt_splitted, 'mtsamples', doc_num)
    
    return mt_splitted, specs
```

```{code} python
quaero, specs_quaero = load_quaero()
e3c, specs_e3c = load_e3c()
casm2, specs_casm2 = load_casm2()
#mt, specs_mt = load_processed_mtsamples()
```

```{code} python
specs_quaero.T
```

```{code} python
specs_e3c.T
```

```{code} python
specs_casm2.T
```

```{code} python
corpus = {'quaero':quaero, 'e3c':e3c, 'casm2':casm2}

for corpa_name, corpa in corpus.items():
    for split_name, split in corpa.items():
        output = f"datasets/{corpa_name}/{split_name}.jsonl"
        save_text_documents(split, output)
```
