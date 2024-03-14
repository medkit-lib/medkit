# Detecting text duplicates

Medkit provides support for detecting duplicates (zones of identical text)
across a set of documents through the {class}`~.preprocessing.DuplicateFinder`
operation, which itself relies on the
[duptextfinder](https://github.com/equipe22/duplicatedZoneInClinicalText/)
library developed at the HEGP.

No optional dependencies are required to use
{class}`~.preprocessing.DuplicateFinder` but it may perform faster if the `ncls`
package is installed:

```console
pip install ncls
```

## Using collections to group documents

The {class}`~.preprocessing.DuplicateFinder` takes as input a list of
{class}`~.core.Collection` of text documents, rather than simply a list of
documents, as other operations. This is because when dealing with large document
bases, it would be very expensive to look for duplicates in all pairs of
documents. So instead, we group together documents more likely to have
duplicates, for instance documents related to the same patient, and only look
for duplicates across documents belonging to the same group.

For the purpose of this tutorial, we have created 2 folders, each folder
containing 2 text files regarding the same patient. The contents of one of the
documents of the first patient were copy-pasted into the other document:

```{code} python
from pathlib import Path

main_dir = Path("data/duplicate_detection")
file_1 = main_dir / "patient_1/a10320aa-2008_04_13.txt"
print(file_1.read_text())
```

```{code} python
file_2 = main_dir / "patient_1/f1d3e530-2008_04_14.txt"
print(file_2.read_text())
```

Let's create a list of
collections, with one collection per patient:

```{code} python
from medkit.core import Collection
from medkit.core.text import TextDocument

# iterate over each subdirectory containing patient files
collections = []
for patient_subdir in sorted(main_dir.glob("*")):
    # create one TextDocument per .txt file
    docs = TextDocument.from_dir(patient_subdir)
    # group them in a Collection
    collection = Collection(text_docs=docs)
    collections.append(collection)
```

## Identifying duplicated zones

Let's now instantiate a duplicate finder and run in on our collections:

```{code} python
from medkit.text.preprocessing import DuplicateFinder

dup_finder = DuplicateFinder(output_label="duplicate")
dup_finder.run(collections)

for collection in collections:
    for doc in collection.text_docs:
        dup_segs = doc.anns.get(label="duplicate")
        for dup_seg in dup_segs:
            print(repr(dup_seg.text))
            attr = dup_seg.attrs.get(label="is_duplicate")[0]
            print(f"{attr.label}={attr.value}")
            print(f"source_doc_id={attr.source_doc_id}")
            print(f"source_spans={attr.source_spans}")
```

As you can see, one duplicated zone has been detected and a segment has been
created to identify the zone. A {class}`~.preprocessing.DuplicationAttribute`
is attached to it, with information about the source document and spans from
which the text was duplicated.

## Using dates to differentiate sources and duplicates

When the {class}`~.preprocessing.DuplicateFinder` encounters 2 identical pieces
of text in 2 different documents, it has to decide which one is the original and
which one is the duplicate. Obviously, we want to consider the text for the
oldest document as the source and the text for the newest as the duplicate.

The default behavior of {class}`~.preprocessing.DuplicateFinder` is to assume
that the text documents are sorted from oldest to newest in the collection.
However, that is not necessarily the case. This is why
{class}`~.preprocessing.DuplicateFinder` also supports retrieving the date from
the document metadata.

Let's rebuild our collection of text documents, adding a `"creation_date"` entry
to the metadata of each doc (that we extract from the filename for the purpose
of the example):

```{code} python
collections = []
for patient_subdir in sorted(main_dir.glob("*")):
    docs = []
    for file in patient_subdir.glob("*.txt"):
        # example file name: 02e0b400-2012_01_29.txt
        # we extract the date from the 2nd part of the base name
        date = file.name.split("-")[1]
        # add the date to the document metadata under the "creation_date" key
        doc = TextDocument(text=file.read_text(), metadata={"creation_date": date})
        docs.append(doc)
    collection = Collection(text_docs=docs)
    collections.append(collection)
```

and let's use that metadata when finding duplicates:

```{code} python
# tell DuplicateFinder to use the "creation_date" metadata to order documents
dup_finder = DuplicateFinder(output_label="duplicate", date_metadata_key="creation_date")
dup_finder.run(collections)
```

Note that the values of the date metadata should be sortable, in a way that
makes sense. For instance using date strings with format "DD-MM-YYYY" rather
than "YYYY-MM-DD" wouldn't work.

## Ignoring duplicated zones

Most of the time, we will probably want to ignore duplicate zones, ie to work on
segments identifying the non-duplicate zones of our documents. This can be
achieved with the `segments_to_output` init parameter of
{class}`~.preprocessing.DuplicateFinder`. By default, it is set to `"dup"`, which means that
segments for duplicate zones will be added to documents. But it can be set
instead to `"nondup"`, in which case segments for non-duplicate zones will be
added to documents[^1].

Let's see an example of how to run a minimalistic NER pipeline on the
non-duplicate zones of our documents:

```{code} python
from medkit.core import DocPipeline, Pipeline, PipelineStep
from medkit.text.segmentation import SentenceTokenizer
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

# create segments for non-duplicate zones with label "nonduplicate"
dup_finder = DuplicateFinder(
    output_label="nonduplicate",
    segments_to_output="nondup",
    date_metadata_key="creation_date",
)

# create a minimalistic NER pipeline
sentence_tok = SentenceTokenizer(output_label="sentence")
matcher = RegexpMatcher(rules=[RegexpMatcherRule(regexp=r"\binsuffisance\s*rénale\b", label="problem")])
pipeline = Pipeline(
    steps=[
        PipelineStep(sentence_tok, input_keys=["raw_text"], output_keys=["sentences"]),
        PipelineStep(matcher, input_keys=["sentences"], output_keys=["entities"]),
    ],
    input_keys=["raw_text"],
    output_keys=["entities"],
)

# use "nonduplicate" segments as input to the NER pipeline
doc_pipeline = DocPipeline(
    pipeline=pipeline,
    labels_by_input_key={"raw_text": ["nonduplicate"]},
)

# run everything
dup_finder.run(collections)
for collection in collections:
    doc_pipeline.run(collection.text_docs)
```

Let's now visualize the annotations of the 2 documents of the first patient:

```{code} python
from spacy import displacy
from medkit.text.spacy.displacy_utils import medkit_doc_to_displacy

doc_1 = collections[0].text_docs[0]
displacy_data = medkit_doc_to_displacy(doc_1)
displacy.render(displacy_data, manual=True, style="ent")
```

```{code} python
doc_2 = collections[0].text_docs[1]
displacy_data = medkit_doc_to_displacy(doc_2)
displacy.render(displacy_data, manual=True, style="ent")
```

As expected, the "insuffisance rénale" entity was only found in the original
report but properly ignored in the more recent report that copy-pasted it.


[^1]: It is also possible to set `segments_to_output` to `"both"` and use the
    value of the attributes to distinguish between duplicate and non-duplicated
    segments.
