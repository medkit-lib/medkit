# Text spans

Here are some examples about usage of span utilities.

```{code} python
from medkit.core.text.span import Span
from medkit.core.text.span_utils import replace, remove, move, extract, insert

raw_text = (
    "Cher M. Dupond,\nJ’ai vu en consultation (à mon cabinet le 2019-02-01) "
    "Bertrand AGITE, né le 2008-02-25,"
    "\n\npour une suspicion de troubles du spectre autistique.\n(-) TDAH.\n"
)
text = raw_text
spans = [Span(0, len(raw_text))]
```

```{code} python
import re

# replace "M." by "M
# `spans` keeps the modifications 
match = re.search(r"M.", text, re.M)
text, spans = replace(text, spans, [match.span()], ["M"])
print(text)
print(spans)
:::

:::{code}
# remove final endline
match = re.search(r"\n\Z", text, re.M)
text, spans = remove(text, spans, [match.span()])

# replace line breaks with spaces
ranges = [m.span() for m in re.finditer(r"\n+", text, re.M)]
text, spans = replace(text, spans, ranges, [" "] * len(ranges))
print(text)
```

```{code} python
# extract sentences
sentences = []
for match in re.finditer(r"[^\.]+\.", text, re.M):
    sentence_text, sentence_spans = extract(text, spans, [match.span()])
    sentences.append((sentence_text, sentence_spans))

text_1, spans_1 = sentences[0]
text_2, spans_2 = sentences[1]
print(text_1)
print(text_2)
```

```{code} python
# move parenthesized text to end in 1st sentence
match = re.search(r" *\((.*)\)", text_1, re.M)
text_1, spans_1 = insert(text_1, spans_1, [len(text_1) - 1], [" ; "])
text_1, spans_1 = move(text_1, spans_1, match.span(1), len(text_1) - 1)
print(text_1)
```

```{code} python
# reformat dates in 1st sentence
matches = list(re.finditer(r"\d{4}-\d{2}-\d{2}", text_1, re.M))
ranges = [m.span() for m in matches]
new_dates = [
    m.group(0)[8:10] + "/" + m.group(0)[5:7] + "/" + m.group(0)[0:4]
    for m in matches
]
text_1, spans_1 = replace(text_1, spans_1, ranges, new_dates)
print(text_1)
```

```{code} python
# replace "(-)" by "negatif" in 2d sentence
match = re.search(r"\(-\)", text_2, re.M)
text_2, spans_2 = replace(text_2, spans_2, [match.span()], ["negatif"])
print(text_2)
```

```{code} python
# find person entity in 1st sentence
match = re.search(r"M [a-zA-Z]+", text_1)
person_text, person_spans = extract(
    text_1, spans_1, [match.span()]
)
```

```{code} python
# find date entities in 1st sentence
dates = []
for match in re.finditer(r"\d{2}/\d{2}/\d{4}", text_1):
    date_text, date_spans = extract(text_1, spans_1, [match.span()])
    dates.append((date_text, date_spans))
```

```{code} python
from medkit.core.text.span_utils import normalize_spans

entities = []

person_spans = normalize_spans(person_spans)
entities.append(("person", person_spans))
for _, date_spans in dates:
    date_spans = normalize_spans(date_spans)
    entities.append(("date", date_spans))
print(entities)
```

```{code} python
from spacy import displacy

entities_data = [
    {"start": span.start, "end": span.end, "label": label}
    for label, spans in entities
    for span in spans
]
entities_data = sorted(entities_data, key=lambda e: e["start"])
data = {"text": raw_text, "ents": entities_data, "uuid": 0}
displacy.render(data, manual=True, style="ent", jupyter=True, minify=True)
```
