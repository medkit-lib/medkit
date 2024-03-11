# Audio Processing

This page lists and explains all components related to audio processing.

For more details about the public API, please refer to {mod}`medkit.audio`.

```{contents} Table of Contents
:depth: 3
```

## Preprocessing Operations

This section provides some information about how to use preprocessing modules for audio.

For more details about public APIs, refer to {mod}`medkit.audio.preprocessing`.

### Downmixer

Refer to {mod}`medkit.audio.preprocessing.downmixer`.

### Power Normalizer

Refer to {mod}`medkit.audio.preprocessing.power_normalizer`.

### Resampler

Refer to {mod}`medkit.audio.preprocessing.resampler`.

::::{important}
{class}`~.audio.preprocessing.resampler.Resampler` requires additional dependencies:

```console
pip install 'medkit-lib[resampler]'
```
::::

## Segmentation Operations

This section lists audio segmentation operations,
which are included in the {mod}`medkit.audio.segmentation` module.

### WebRTC Voice Detector

Refer to {mod}`medkit.audio.segmentation.webrtc_voice_detector`.

### Speaker Detector

Refer to {mod}`medkit.audio.segmentation.pa_speaker_detector`.

::::{important}
{class}`~.audio.segmentation.pa_speaker_detector.PASpeakerDetector` requires additional dependencies:

```console
pip install 'medkit[pa-speaker-detector]'
```
::::

## Audio Transcription

This section lists operations used to perform audio transcription.
They are part of the {mod}`medkit.audio.transcription` module.

{class}`~.audio.transcription.DocTranscriber` is the operation handling the
transformation of {class}`~.core.audio.AudioDocument` instances into
{class}`~.audio.transcription.TranscribedTextDocument` instances.

The actual conversion from text to audio is delegated to operation complying
with the {class}`~.audio.transcription.TranscriptionOperation` protocol.
{class}`~.audio.transcription.hf_transcriber.HFTranscriber` and
{class}`~.audio.transcription.sb_transcriber.SBTranscriber` are implementations
of {class}`~.audio.transcription.TranscriptionOperation`,
which use HuggingFace transformer and SpeechBrain models respectively.

### DocTranscriber

Refer to {mod}`medkit.audio.transcription.doc_transcriber`.

### TranscribedTextDocument

Refer to {mod}`medkit.audio.transcription.transcribed_text_document`.

### HFTranscriber

Refer to {mod}`medkit.audio.transcription.hf_transcriber`.

::::{important}
{class}`~.audio.transcription.hf_transcriber.HFTranscriber` requires additional dependencies:

```
pip install 'medkit-lib[hf-transcriber]'
```
::::

### SBTranscriber

Refer to {mod}`medkit.audio.transcription.sb_transcriber`.

::::{important}
{class}`~.audio.transcription.sb_transcriber.SBTranscriber` requires additional dependencies:

```
pip install 'medkit-lib[sb-transcriber]'
```
::::

## Metrics

Module {mod}`medkit.audio.metrics` provides components to evaluate audio annotations.
