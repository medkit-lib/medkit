# Evaluation

```{code} python
from medkit.io.medkit_json import load_text_documents
from medkit.core.text import TextDocument
from pathlib import Path
from tqdm import tqdm
from medkit.text.ner import UMLSMatcher
from medkit.text.metrics.ner import SeqEvalEvaluator
import pandas as pd
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher
import torch
from medkit.core import Pipeline, DocPipeline, PipelineStep
from medkit.text.postprocessing import filter_overlapping_entities, DocumentSplitter
from spacy_llm.util import assemble
import os 
from medkit.core.text import Entity, Span
from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy
from IPython.display import display, HTML
from typing import List 

DEVICE = 0 if torch.cuda.is_available() else -1


class GPT_NER:
    def __init__(self, config, _open_ai_key, _name):
        os.environ["OPENAI_API_KEY"] = _open_ai_key
        self.nlp = assemble(config)
        self.description = self.Description(_name)

    def run(self, _raw_segment : []):
        doc = self.nlp(_raw_segment[0].text)
        gpt_remapping = { "Anatomie": "ANAT", "Médicament": "CHEM", "Appareil": "DEVI", "Pathologie": "DISO", "Région": "GEOG", "Organisme": "LIVB", "Objet": "OBJC", "Phénomène": "PHEN", "Physiologie": "PHYS", "Procédure": "PROC"}
        predicted_entities = [Entity(label=gpt_remapping[ent.label_], text=ent.text, spans=[Span(ent.start_char, ent.end_char)]) for ent in doc.ents]

        return predicted_entities
    
    class Description:
        def __init__(self, _name):
            self.name = _name

#Mets en forme les résultats renvoyé par SeqEvalEvaluator
def results_to_df(_results, _title):
    results_list = list(_results.items())
    arranged_results = {"Entities": ['P', 'R', 'F1'], 
                        "all": [round(results_list[i][1], 2) for i in [0, 1, 2]]}
    accuracy = round(results_list[4][1], 2)

    for i in range(5, len(results_list), 4):
        key = results_list[i][0][:-10]
        arranged_results[key] = [round(results_list[n][1], 2) for n in [i, i + 1, i + 2]]

    df = pd.DataFrame(arranged_results, index=[f"{_title} ({accuracy})", '', '']).T
    return df

#Evalue les annotations de plusieurs outils de NER sur les documents fournis
def eval(_docs, open_ai_key="", _labels_remapping = {"ANAT": "ANAT", "CHEM": "CHEM", "DEVI": "DEVI", "DISO": "DISO", "GEOG": "GEOG",
                                                     "LIVB": "LIVB", "OBJC": "OBJC", "PHEN": "PHEN", "PHYS": "PHYS", "PROC": "PROC",}):

    ners = []
    ners.append(UMLSMatcher(umls_dir=Path.home() / "src/UMLS", cache_dir=".umls_cache", language="FRE", semgroups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'], lowercase=True, normalize_unicode=True, name="umls_matcher", output_labels_by_semgroup=_labels_remapping))
    ners.append(HFEntityMatcher(model="Thibeb/CamemBert_bio_generalized", name="drbert_matcher", device=DEVICE))
    ners.append(HFEntityMatcher(model="Thibeb/DrBert_generalized", name="camembert_matcher", device=DEVICE))
    if open_ai_key != "" : ners.append(GPT_NER("config_gpt.cfg", open_ai_key,_name='ChatGPT-3.5-turbo'))

    ner_evaluator = SeqEvalEvaluator(return_metrics_by_label=True, average='weighted', labels_remapping=_labels_remapping) 

    dfs = []

    for ner in ners:
        predicted_entities = [ner.run([doc.raw_segment]) for doc in tqdm(_docs)]
        results = ner_evaluator.compute(_docs, predicted_entities)
        dfs.append(results_to_df(_results=results, _title=ner.description.name))
        
    return pd.concat(dfs, axis=1)

#Affiche les annotation de plusieurs outils de NER sur le document fourni
def test(_doc, open_ai_key="", _labels_remapping={"ANAT": "ANAT", "CHEM": "CHEM", "DEVI": "DEVI", "DISO": "DISO", "GEOG": "GEOG",
                                                  "LIVB": "LIVB", "OBJC": "OBJC", "PHEN": "PHEN", "PHYS": "PHYS", "PROC": "PROC",}):
    
    ners = []
    ners.append(UMLSMatcher(umls_dir=Path.home() / "src/UMLS", cache_dir=".umls_cache", language="FRE", semgroups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'], lowercase=True, normalize_unicode=True, name="UMLS matcher", output_labels_by_semgroup=_labels_remapping))
    ners.append(HFEntityMatcher(model="Thibeb/CamemBERT_bio_generalized", name="DrBERT matcher", device=DEVICE))
    ners.append(HFEntityMatcher(model="Thibeb/DrBERT_generalized", name="CamemBERT-bio matcher", device=DEVICE))
    if open_ai_key != "" : ners.append(GPT_NER("config_gpt3.cfg", open_ai_key,_name='ChatGPT-3.5-turbo'))
    if open_ai_key != "" : ners.append(GPT_NER("config_gpt4.cfg", open_ai_key,_name='ChatGPT-4'))

    annotated_docs = {'Original':_doc}
    for ner in ners:
        annotated_docs[ner.description.name] = TextDocument(text=_doc.text)
        entities = ner.run([_doc.raw_segment])
        for ent in entities:
            annotated_docs[ner.description.name].anns.add(ent)

    html_datas = [f'<h1>{name}</h1>{displacy.render(medkit_doc_to_displacy(doc), manual=True, style="ent", jupyter=False)}' for name, doc in annotated_docs.items()]
    display(HTML("".join(html_datas)))
```

```{code} python
#Charge une partie du split de test du corpus QUAERO déja pre-processé
docs = list(load_text_documents("datasets/quaero/test.jsonl"))[:100]
```

```{code} python
#Evalue plusieurs outils de NER et renvoie un tableau de comparaison
eval(docs)
```

```{code} python
#Affiche les annotations de différent outils de NER sur un document du split
test(docs[12], open_ai_key="")
```
