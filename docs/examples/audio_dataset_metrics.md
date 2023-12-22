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

# Computing metrics on an audio dataset

This demo shows how to compute diarization and transcription metrics on an audio
dataset such as [simsamu](https://huggingface.co/datasets/medkit/simsamu)

Download the dataset from the HuggingFace hub:

```{code-cell} ipython3
:tags: [skip-execution]
import huggingface_hub as hf_hub
from medkit.io import SRTInputConverter

simsamu_dir = hf_hub.snapshot_download("medkit/simsamu", repo_type="dataset")
```

Load the `.m4a` audio files into audio documents, as well as reference
diarization and transcription annotated documents from corresponding `.rttm` and
`.srt` files:

```{code-cell} ipython3
:tags: [skip-execution]
from pathlib import Path
from medkit.core.audio import AudioDocument
from medkit.io.rttm import RTTMInputConverter
from medkit.io.srt import SRTInputConverter

# init input converters for .rttm and .srt files
rttm_converter = RTTMInputConverter(turn_label="speech", speaker_label="speaker")
srt_converter = SRTInputConverter(turn_segment_label="speech", transcription_attr_label="transcription")

docs = []
ref_docs_diar = []
ref_docs_transcript = []

for rec_dir in sorted(Path(simsamu_dir).glob("*"))[:4]:
    # iterate only on subdirs
    if not rec_dir.is_dir():
        continue
    
    # locate audio, .rttm and .srt files
    m4a_file = next(rec_dir.glob("*.m4a"))
    rttm_file = next(rec_dir.glob("*.rttm"))
    srt_file = next(rec_dir.glob("*.srt"))

    # convert m4a to wav with ffmpeg
    wav_file = m4a_file.with_suffix(".wav")
    if not wav_file.exists():
        !ffmpeg -i {m4a_file} -acodec pcm_s16le -ac 1 -ar 16000 {wav_file}

    # load empty audio doc
    doc = AudioDocument.from_file(wav_file)
    docs.append(doc)
    # load reference audio doc with diarization annotations
    ref_doc_diar = rttm_converter.load_doc(rttm_file=rttm_file, audio_file=wav_file)
    ref_docs_diar.append(ref_doc_diar)
    # load reference audio doc with transcription annotations
    ref_doc_transcript = srt_converter.load_doc(srt_file=srt_file, audio_file=wav_file)
    ref_docs_transcript.append(ref_doc_transcript)
```

Initialize the diarization operation with the [simsamu-diarization pipeline](https://huggingface.co/medkit/simsamu-diarization)

```{code-cell} ipython3
:tags: [skip-execution]
import torch
from medkit.audio.segmentation.pa_speaker_detector import PASpeakerDetector

device = 0 if torch.cuda.is_available() else -1

speaker_detector = PASpeakerDetector(
    model="medkit/simsamu-diarization",
    output_label="speech",
    min_nb_speakers=1,
    max_nb_speakers=2,
    device=device,
    segmentation_batch_size=10,
    embedding_batch_size=10,
)
```

Initialize the transcription operation with the [simsamu-transcription model](https://huggingface.co/medkit/simsamu-transcription):

```{code-cell} ipython3
:tags: [skip-execution]
from medkit.audio.transcription.sb_transcriber import SBTranscriber

transcriber = SBTranscriber(
    model="medkit/simsamu-transcription",
    needs_decoder=False,
    output_label="transcription",
    device=device,
    batch_size=10,
)
```

Diarize and transcribe all documents:

```{code-cell} ipython3
:tags: [skip-execution]
from tqdm import tqdm

# list of list of segments, per document
# (this structure is needed to compute the metrics)
all_speech_segs = []

for doc in tqdm(docs):
    speech_segs = speaker_detector.run([doc.raw_segment])
    transcriber.run(speech_segs)
    all_speech_segs.append(speech_segs)
```

Compute the DER (Diarization Error Rate):

```{code-cell} ipython3
:tags: [skip-execution]
from medkit.audio.metrics.diarization import DiarizationEvaluator

diarization_evaluator = DiarizationEvaluator(
    turn_label="speech",
    speaker_label="speaker",
    collar=0.5,
)

results = diarization_evaluator.compute(ref_docs_diar, all_speech_segs)
print(f"der={results.der:.2%}")
```

```
der=13.45%
```

Compute the WER (Word Error Rate) and CER (Character Error Rate):

```{code-cell} ipython3
:tags: [skip-execution]
from medkit.audio.metrics.transcription import TranscriptionEvaluator

transcription_evaluator = TranscriptionEvaluator(
    speech_label="speech",
    transcription_label="transcription",
)

results = transcription_evaluator.compute(ref_docs_transcript, all_speech_segs)
print(f"wer={results.wer:.2%}, cer={results.cer:.2%}")
```

```
wer=20.77%, cer=15.13%
```

Note that running the transcription operation on the reference speech turns
rather than those returned by the diarization operation will give lower WER and
CER values (around 15% and 9%).