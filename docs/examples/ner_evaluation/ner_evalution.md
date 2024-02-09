---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

```{code-cell} ipython3
from medkit.io.medkit_json import load_text_documents
from medkit.core.text import TextDocument
from medkit.text.segmentation import SentenceTokenizer
from pathlib import Path
from tqdm import tqdm
from medkit.text.ner import UMLSMatcher
from medkit.text.metrics.ner import SeqEvalEvaluator
import pandas as pd
from glob import glob
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher
import torch
from medkit.core import Pipeline, DocPipeline, PipelineStep
from medkit.text.postprocessing import filter_overlapping_entities, DocumentSplitter
from spacy_llm.util import assemble
import os 
from medkit.core.text import Entity, Span
from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy
import random
from IPython.display import display, HTML
import pandas as pd

DEVICE = 0 if torch.cuda.is_available() else -1

def results_to_df(_results, _title):
    results_list = list(_results.items())
    arranged_results = {}
    arranged_results["Entities"] = ['P','R','F1']
    arranged_results["all"] = [round(results_list[i][1],2) for i in [0,1,2]]
    accuracy = round(results_list[4][1],2)
    for i in range(5,len(results_list), 4):
        key = results_list[i][0][:-10]
        arranged_results[key] = [round(results_list[n][1],2) for n in [i,i+1,i+2]]
    df = pd.DataFrame(arranged_results, index=[f"{_title} ({accuracy})",'','']).T
    return df

def LLMNER(_text, _nlp):
    doc = _nlp(_text)
    
    predicted_entities = []

    gpt_remapping = {
    "Anatomie": "ANAT",
    "Médicament": "CHEM",
    "Appareil": "DEVI",
    "Pathologie": "DISO",
    "Région": "GEOG",
    "Organisme": "LIVB",
    "Objet": "OBJC",
    "Phénomène": "PHEN",
    "Physiologie": "PHYS",
    "Procédure": "PROC",
    }

    for ent in doc.ents:
        predicted_entities.append(Entity(
            label=gpt_remapping[ent.label_],
            text=ent.text,
            spans=[Span(ent.start_char, ent.end_char)]
        ))

    return predicted_entities

def eval(_docs, open_ai_key = "", _labels_remapping = {
        "ANAT": "ANAT", "CHEM": "CHEM", "DEVI": "DEVI", "DISO": "DISO", "GEOG": "GEOG",
        "LIVB": "LIVB", "OBJC": "OBJC", "PHEN": "PHEN", "PHYS": "PHYS", "PROC": "PROC",}):

    use_gpt = True if open_ai_key != "" else False
    
    if use_gpt:
        os.environ["OPENAI_API_KEY"] = open_ai_key
        nlp = assemble("config_gpt.cfg")

    #Loading entity matchers
    umls_matcher = UMLSMatcher(
        umls_dir=Path.home() / "src/UMLS",
        cache_dir=".umls_cache",
        language="FRE",
        semgroups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
        lowercase=True,
        normalize_unicode=True,
        name="umls_matcher",
        output_labels_by_semgroup=_labels_remapping
    )

    drbert_matcher = HFEntityMatcher(model="Thibeb/CamemBert_bio_generalized", name="drbert_matcher", device=DEVICE)
    camembert_matcher = HFEntityMatcher(model="Thibeb/DrBert_generalized", name="camembert_matcher", device=DEVICE)

    #Prediction
    ners = [umls_matcher, drbert_matcher, camembert_matcher]
    predicted_entities = {}
    for ner in ners:
        predicted_entities[ner.description.name] = []
    if use_gpt : predicted_entities['GPT-3.5-turbo'] = []

    # Predict entites for each doc for each NER tool
    for doc in tqdm(_docs):
        for ner in ners:
            entities = ner.run([doc.raw_segment])
            predicted_entities[ner.description.name].append(entities)
        if use_gpt:
            entities = LLMNER(doc.text, nlp)
            predicted_entities['GPT-3.5-turbo'].append(entities)
    
    ner_evaluator = SeqEvalEvaluator(return_metrics_by_label=True, average='weighted', labels_remapping=_labels_remapping) 

    # Compute NER metrics for each NER tool
    dfs = []
    for ner in ners:
        results = ner_evaluator.compute(_docs, predicted_entities[ner.description.name])
        dfs.append(results_to_df(_results=results, _title=ner.description.name))
    if use_gpt:
        results = ner_evaluator.compute(_docs, predicted_entities['GPTNER'])
        dfs.append(results_to_df(_results=results, _title='GPTNER'))

    return pd.concat(dfs, axis=1)

def test(_doc, open_ai_key = "", _labels_remapping = {
        "ANAT": "ANAT", "CHEM": "CHEM", "DEVI": "DEVI", "DISO": "DISO", "GEOG": "GEOG",
        "LIVB": "LIVB", "OBJC": "OBJC", "PHEN": "PHEN", "PHYS": "PHYS", "PROC": "PROC",}):
    
    use_gpt = True if open_ai_key != "" else False
    
    if use_gpt:
        os.environ["OPENAI_API_KEY"] = open_ai_key
        nlp = assemble("config_gpt.cfg")

    #Loading entity matchers
    umls_matcher = UMLSMatcher(
        umls_dir=Path.home() / "src/UMLS",
        cache_dir=".umls_cache",
        language="FRE",
        semgroups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
        lowercase=True,
        normalize_unicode=True,
        name="umls_matcher",
        output_labels_by_semgroup=_labels_remapping
    )

    drbert_matcher = HFEntityMatcher(model="Thibeb/CamemBert_bio_generalized", name="drbert_matcher", device=DEVICE)
    camembert_matcher = HFEntityMatcher(model="Thibeb/DrBert_generalized", name="camembert_matcher", device=DEVICE)

    #Prediction
    ners = [umls_matcher, drbert_matcher, camembert_matcher]
    annotated_docs = {}

    # Predict entites for each doc for each NER tool
    for ner in ners:
        entities = ner.run([_doc.raw_segment])
        annotated_doc = TextDocument(text=_doc.text)
        for ent in entities:
            annotated_doc.anns.add(ent)
        annotated_docs[ner.description.name] = annotated_doc
        
    if use_gpt:
        entities = LLMNER(_doc.text, nlp)
        annotated_doc = TextDocument(text=_doc.text)
        for ent in entities:
            annotated_doc.anns.add(ent)
        annotated_docs['GPT-3.5-turbo'] = annotated_doc

    html_datas = []

    for ner, doc in annotated_docs.items():
        displacy_data = medkit_doc_to_displacy(doc)
        html_data = displacy.render(displacy_data, manual=True, style="ent", jupyter=False)
        html_datas.append(html_data)

    display(HTML("".join(html_datas)))  
```

```{code-cell} ipython3
docs = list(load_text_documents("datasets/quaero/test.jsonl"))[14:25]

eval(docs[:10])
```

```{code-cell} ipython3
def eval(_docs, _labels_remapping, )
```
