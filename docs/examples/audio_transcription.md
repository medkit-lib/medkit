---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Audio transcription

This demo shows how to transcribe an audio document and then perform text
operations on it.

## Init audio document

Instantiate an {class}`~.core.audio.AudioDocument` with a
{class}`~.core.audio.FileAudioBuffer`:

```{code-cell} ipython3
from pathlib import Path
import IPython.display
from medkit.core.audio import AudioDocument, FileAudioBuffer

audio_file = Path("input/voice.ogg")
audio_doc = AudioDocument(audio=FileAudioBuffer(audio_file))

IPython.display.Audio(data=audio_doc.audio.read(), rate=audio_doc.audio.sample_rate)
```

## Voice detection

Prepare pipeline to perform voice detection on audio documents, using a
{class}`~.audio.preprocessing.Downmixer` chained with a
{class}`~.audio.segmentation.webrtc_voice_detector.WebRTCVoiceDetector` (you can
also use other segmentation operations such as
{class}`~.audio.segmentation.pa_speaker_detector.PASpeakerDetector` ):

```{code-cell} ipython3
from medkit.core import Pipeline, PipelineStep, DocPipeline
from medkit.audio.preprocessing import Downmixer
from medkit.audio.segmentation.webrtc_voice_detector import WebRTCVoiceDetector

# init operations
downmixer = Downmixer(output_label="mono")
voice_detector = WebRTCVoiceDetector(output_label="voice")

# put them in a pipeline
audio_pipeline = Pipeline(
    steps=[
        PipelineStep(
            downmixer,
            input_keys=["full_audio"],
            output_keys=["full_mono_audio"],
        ),
        PipelineStep(
            voice_detector,
            input_keys=["full_mono_audio"],
            output_keys=["voice_segs"],
        ),
    ],
    input_keys=["full_audio"],
    output_keys=["voice_segs"],
)

# wrap pipeline in doc-level pipeline
audio_doc_pipeline = DocPipeline(audio_pipeline)
```

Run voice detection on audio document:

```{code-cell} ipython3
audio_doc_pipeline.run([audio_doc])
for seg in audio_doc.anns.get(label="voice"):
    print(f"label={seg.label}, span={seg.span}")
```

## Transcription

Prepare a {class}`~.audio.transcription.DocTranscriber` that will convert audio
documents to text documents, using
{class}`~.audio.transcription.hf_transcriber.HFTranscriber` as the actual audio
transcriber creating text segments from audio segments (you can also use other
transcription operations such as
{class}`~.audio.transcription.sb_transcriber.SBTranscriber`):

```{code-cell} ipython3
:tags: [skip-execution]
from medkit.audio.transcription import DocTranscriber
from medkit.audio.transcription.hf_transcriber import HFTranscriber

transcriber = HFTranscriber(
    model="openai/whisper-small",
    language="english",
    add_trailing_dot=False,
    capitalize=False,
)
doc_transcriber = DocTranscriber(
    input_label="voice",
    output_label="transcription",
    transcription_operation=transcriber,
)
```

Transcribe audio document:

```{code-cell} ipython3
:tags: [skip-execution]
transcribed_doc = doc_transcriber.run([audio_doc])[0]
print(f"fulltext={transcribed_doc.text!r}", end="\n\n")
for seg in transcribed_doc.anns.get(label="transcription"):
    print(f"label={seg.label}, text={seg.text!r}")
```

```
fulltext=' I have headaches.\n I also have high blood pressure.'

label=transcription, text=' I have headaches.'
label=transcription, text=' I also have high blood pressure.'
```

## Entity matching on text

Run text entity matching on transcribed document:

```{code-cell} ipython3
:tags: [skip-execution]
from medkit.core.text import TextDocument
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule

rules = [
    RegexpMatcherRule(label="problem", regexp=r"\bheadaches?\b"),
    RegexpMatcherRule(label="problem", regexp=r"\bhigh\s+blood\s+pressure\b"),
]
matcher = RegexpMatcher(rules)
text_pipeline = Pipeline(
    steps=[PipelineStep(matcher, input_keys=["full_text"], output_keys=["entities"])],
    input_keys=["full_text"],
    output_keys=["entities"]
)
text_doc_pipeline = DocPipeline(
    text_pipeline,
    labels_by_input_key={"full_text": [TextDocument.RAW_LABEL]},
)
text_doc_pipeline.run([transcribed_doc])
```

Locate matched entities in original audio:

```{code-cell} ipython3
:tags: [skip-execution]
entities = transcribed_doc.anns.get_entities()

for entity in entities:
    print(f"label={entity.label}, text={entity.text!r}")
    audio_spans = transcribed_doc.get_containing_audio_spans(entity.spans)
    print(f"audio_spans={audio_spans}", end="\n\n")

    audio = audio_doc.audio.trim_duration(audio_spans[0].start, audio_spans[0].end)
    IPython.display.display(IPython.display.Audio(data=audio.read(), rate=audio.sample_rate))
```

```{code-cell} ipython3
:tags: [remove-input]
# hardcoded display of audio spans to workaround
# the fact that cells are not executed
print("label=problem, text='headaches'")
entity_1_audio = audio_doc.audio.trim_duration(0.99, 2.73)
IPython.display.display(IPython.display.Audio(data=entity_1_audio.read(), rate=entity_1_audio.sample_rate))

print("label=problem, text='high blood pressure'")
entity_2_audio = audio_doc.audio.trim_duration(6.0, 8.73)
IPython.display.display(IPython.display.Audio(data=entity_2_audio.read(), rate=entity_2_audio.sample_rate))
```
