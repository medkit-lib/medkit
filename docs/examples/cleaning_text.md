---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Cleaning text with a predefined operation

+++

Medkit allows us to transform and clean up text without destroying the original text. We could, for example, implement a set of clean-up steps within the `run` method of an operation to pre-process raw text.

In this example, we will use a predefined {class}`~medkit.text.preprocessing.eds_cleaner.EDSCleaner` operation to show how a cleaning process works in medkit. This operation is inspired by french documents with formatting problems given previous conversion processes. 

## Loading a text to clean

Consider the following document:

```{code-cell} ipython3
# You can download the file available in source code
# !wget https://raw.githubusercontent.com/medkit-lib/medkit/main/docs/examples/input/text/text_to_clean.txt

from pathlib import Path
from medkit.core.text import TextDocument

doc = TextDocument.from_file(Path("./input/text/text_to_clean.txt"))
print(doc.text)
```
As we note, the text has:
- additional spaces;
- multiple newlines characters;
- long parentheses and numbers in English format.

This complicates text segmentation of the text, it may be a good idea to clean up the text before segmenting or creating annotations.
  
## Using EDSCleaner operation

As mentioned before, you can create your own custom cleanup operation. In this case, we use the predefined operation for a french document (coming from the EDS) to format the document.

The main idea is to transform the `raw_segment` and keep track of the modifications made by the operation. That segment is defined using the span of the text.

**A span in medkit**
> In medkit the span of an annotation is a list of simple spans {class}`~medkit.core.text.Span`  or modified spans {class}`~medkit.core.text.ModifiedSpan`. With this mechanism, we keep track of the modifications and can return to the original version whenever we want.

The `EDSCleaner` is configurable, we initialize `keep_endlines=True` to facilitate the visualization. Otherwise, the output segment would be a plain text with no newlines `(\n)` characters. 


```{code-cell} ipython3
from medkit.text.preprocessing import EDSCleaner

eds_cleaner = EDSCleaner(keep_endlines=True)
raw_segment = doc.raw_segment
clean_segment = eds_cleaner.run([raw_segment])[0]
print(clean_segment.text)

```

The class works on `Segments`. In the `run` method it performs several operations to delete or change characters of interest. By default, it performs these operations:

* Changes points between uppercase letters to spaces
* Changes points between numbers to commas
* Deletes multiple newline characters.
* Deletes multiple whitespaces. 

```{note}
There are two special operations that process parentheses and dots near French keywords such as Dr., Mme. and others. To enable/disable these operations you can use `handle_parentheses_eds` and `handle_points_eds`.
```

## Extract text from the clean text

Now that we have a **clean segment**, we can run an operation on the new segment. We can detect the sentences, for example.


```{code-cell} ipython3
from medkit.text.segmentation import SentenceTokenizer

sentences = SentenceTokenizer().run([clean_segment])
for sent in sentences:
  print(f"{sent.text!r}")
```

**A created sentence in detail**

The span of each generated sentence contains the modifications made by *eds_cleaner* object. Let's look at the second sentence:  


```{code-cell} ipython3
sentence = sentences[1]
print(f"text={sentence.text!r}")
print("spans=\n","\n".join(f"{sp}" for sp in sentence.spans))
```

The sentence starts with the character `M` (index 56), followed by a point `.` which has been replaced by a space (index 57). Then, the whole text up to the newline character has not been modified, so it corresponds to the original span (index 58 to 110). Each modification is stored by `ModifiedSpan` objects, until the end of the sentence, the character index 177.

## Displaying in the original text

Since the sentence contains the information from the original spans, it will always be possible to go back and display the information in the raw text. 

To get the original spans, we can use {func}`~medkit.core.text.span_utils.normalize_spans`. Next, we can extract the raw text using {func}`~medkit.core.text.span_utils.extract`. 

```{code-cell} ipython3
from medkit.core.text.span_utils import normalize_spans, extract

spans_sentence = normalize_spans(sentence.spans)
ranges = [(s.start, s.end) for s in spans_sentence]
extracted_text, spans = extract(raw_segment.text,raw_segment.spans,ranges)
print(f"- Sentence in the ORIGINAL version:\n \"{extracted_text}\"")
```

That's how an operation transforms text and extracts information without losing the raw text.  

```{seealso}
For further information on the utilities used in this class, see {class}`~medkit.core.text.utils`. 
To see more examples of span operations [here](../examples/spans)

```
