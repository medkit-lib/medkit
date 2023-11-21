# Audio operations

This page lists all components related to audio processing.



:::{note}
For more details about all sub-packages, refer to
{mod}`medkit.audio`.
:::

## Pre-processing operations

This section provides some information about how to use preprocessing modules
for audio.

:::{note}
For more details about public APIs, refer to {mod}`medkit.audio.preprocessing`.
:::

### Downmixer

For more details, refer to {mod}`medkit.audio.preprocessing.downmixer`.

### Power normalizer

For more details, refer to {mod}`medkit.audio.preprocessing.power_normalizer`.

### Resampler

:::{important}
{class}`~.audio.preprocessing.resampler.Resampler` needs additional dependencies
that can be installed with `pip install medkit-lib[resampler]`
:::

For more details, refer to {mod}`medkit.audio.preprocessing.resampler`.

## Segmentation operations

This section lists audio segmentation operations. They are part of the
{mod}`medkit.audio.segmentation` module.


### WebRTC voice detector

For more details, refer to
{mod}`medkit.audio.segmentation.webrtc_voice_detector`.

### Pyannote speaker detector

:::{important}
{class}`~.audio.segmentation.pa_speaker_detector.PASpeakerDetector` is an experimental feature.
It depends on a version of pyannote-audio that [is not released yet](https://github.com/pyannote/pyannote-audio/issues/1460) on PyPI.
:::

To install it, you may use the `JSALT2023` tag :

```
pip install https://github.com/pyannote/pyannote-audio/archive/refs/tags/JSALT2023.tar.gz
```

For more details, refer to {mod}`medkit.audio.segmentation.pa_speaker_detector`.

## Audio Transcription

This section lists operations and other components to use to perform audio
transcription.
They are part of the {mod}`medkit.audio.transcription` module.

{class}`~.audio.transcription.DocTranscriber` is the operation handling the
transformation of {class}`~.core.audio.AudioDocument` instances into
{class}`~.audio.transcription.TranscribedTextDocument` instances (subclass of
{class}`~.core.text.TextDocument`).

The actual conversion from text to audio is delegated to operation complying
with the {class}`~.audio.transcription.TranscriptionOperation` protocol.
{class}`~.audio.transcription.hf_transcriber.HFTranscriber` and
{class}`~.audio.transcription.sb_transcriber.SBTranscriber` are implementations
of {class}`~.audio.transcription.TranscriptionOperation`, allowing to use
HuggingFace transformer models and speechbrain models respectively.

### DocTranscriber

For more details, refer to {mod}`medkit.audio.transcription.doc_transcriber`.

### TranscribedTextDocument

For more details, refer to {mod}`medkit.audio.transcription.transcribed_text_document`.

### HFTranscriber

:::{important}
{class}`~.audio.transcription.hf_transcriber.HFTranscriber` needs additional
dependencies that can be installed with 
`pip install medkit-lib[hf-transcriber]`
:::

For more details, refer to
{mod}`medkit.audio.transcription.hf_transcriber`.

### SBTranscriber

:::{important}
{class}`~.audio.transcription.sb_transcriber.SBTranscriber` needs additional
dependencies that can be installed with
`pip install medkit-lib[sb-transcriber]`
:::

For more details, refer to
{mod}`medkit.audio.transcription.sb_transcriber`.

# Metrics

The module {mod}`medkit.audio.metrics` provides components to evaluate audio annotations.
