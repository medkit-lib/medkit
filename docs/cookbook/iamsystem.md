# IAMSystem Matcher

+++

This tutorial will show an example of iamsystem matcher operation usage.

+++

## Loading a text document

For beginners, let's create a medkit text document from the following text.

```{code} python
from medkit.core.text import TextDocument

text = """Le patient présente une asténie de grade 2 et une anémie de grade 3. 
Atteinte du poumon gauche et droit. Il est traité par chimiothérapie. 
Son père est décédé d'un cancer du poumon. Il n'a pas de vascularite."""

doc = TextDocument(text=text)
```

The full raw text can be accessed through the `text` attribute:

```{code} python
print(doc.text)
```

## Processing raw text before using iamsystem matcher

Before using entity matcher, we want to split the raw text in sentences, and then detect negation and family context on these sentences.

### Initializing the operations

First, let's configure the three text operations.

```{code} python
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
from medkit.text.context import NegationDetector, NegationDetectorRule, FamilyDetector, FamilyDetectorRule

sent_tokenizer = SentenceTokenizer(
output_label="sentence",
punct_chars=[".", "?", "!", "\n"],
)
neg_detector = NegationDetector(output_label="is_negated")
fam_detector = FamilyDetector(output_label="family")
```

### Running the operations

Now, let's run the operations.

```{code} python
sentences = sent_tokenizer.run([doc.raw_segment])
neg_detector.run(sentences)
fam_detector.run(sentences)

print(f"Number of detected sentences: {len(sentences)}\n")

for sentence in sentences:
    print(f"text = {sentence.text!r}")
    print(f"label = {sentence.label}")
    print(f"is_negated = {sentence.attrs.get(label='is_negated')}")
    print(f"family = {sentence.attrs.get(label='family')}")
    print(f"spans = {sentence.spans}\n")
```

As you can see, we have detected 5 sentences. 
By running negation and family context operations, each sentence is a medkit segment which contains additional attributes for these contexts.

For example, the sentence `Son père est décédé d'un cancer du poumon` contains a `family` context attribute and its value is set to `True` because `père` has been detected.

In the same manner, the sentence `Il n'a pas de vascularite` contains a negation attribute which value is `True`, that means that the sentence is considered as negative.

## Using iamsystem matcher for detecting entities

Let's configure the iam system matcher
(cf. [iamsystem official documentation](https://iamsystem-python.readthedocs.io/en/latest/)).

```{code} python
from medkit.text.ner.iamsystem_matcher import MedkitKeyword

from iamsystem import Matcher
from iamsystem import ESpellWiseAlgo

# Defining a keyword for searching "poumon gauche" and tag this entity as
# "anatomy" with normalization information of the detected entity.

medkit_keyword_1 = MedkitKeyword(
                        label="poumon gauche", 
                        kb_id="M001", kb_name="manual",
                        ent_label="anatomy"
                    )
                    
# Defining a keyword for searching "vascularite" and tag this entity as
# "disorder" with normalization information of the detected entity.

medkit_keyword_2 = MedkitKeyword(
                        label="vascularite",
                        kb_id="M002", kb_name="manual",
                        ent_label="disorder")

keywords_list = [medkit_keyword_1, medkit_keyword_2]

# Configuring matcher
matcher = Matcher.build(
            keywords=keywords_list,
            spellwise=[dict(measure=ESpellWiseAlgo.LEVENSHTEIN, max_distance=1, min_nb_char=5)],
            stopwords=["et"],
            w=2
)
```

In this example, we have defined two keywords then configured matcher with:
* the list of keywords to search for : `keywords_list`
* the [Levenshtein spellwise algorithm](https://iamsystem-python.readthedocs.io/en/latest/fuzzy.html#id1)
* a list of words to ignore in the detection : `stopwords`
* a context window `w` to determine how much discontinuous the sequence of tokens can be.


Now, let's configure and run our medkit operation : {class}`~.text.ner.iamsystem_matcher.IAMSystemMatcher`.

```{code} python
from medkit.text.ner.iamsystem_matcher import IAMSystemMatcher

# Configuring medkit operation with iam system matcher and
# tell operation to propagate negation and family context attributes
# from sentences to detected entities
iam = IAMSystemMatcher(matcher = matcher, attrs_to_copy=["is_negated", "family"])

# Run the operation
entities = iam.run(sentences)

print(f"Number of detected entities: {len(entities)}\n")

for entity in entities:
    doc.anns.add(entity)

    print(f"text = {entity.text!r}")
    print(f"label = {entity.label}")
    print(f"normalization = {entity.attrs.get_norms()}")
    print(f"is_negated = {entity.attrs.get(label='is_negated')}")
    print(f"family = {entity.attrs.get(label='family')}")
    print(f"spans = {entity.spans}\n")
```
