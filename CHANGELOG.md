# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.13.0 (2024-02-05)

### Added

- Add nlstruct-based entity matcher

### Changed

- Improve robustness of PASpeakerDetector
- Allow to specify model output language with HFTranscriber

### Fixed

- Use link to new repository
- When parsing BRAT, preserve leading space in entities
- Replace unidecode by anyascii

## 0.12.0 (2023-11-28)

### Added

- Document attributes are now supported (both for text and audio) and are added/accessed the same way as annotations attributes
- Brat Input and Output converters can now load and save UMLS CUIs stored in notes
- new from_dir()/from_file() helper methods added to TextDocument/AudioDocument
- new text classification, audio diarization and audio transcription metrics

### Changed

- the Trainer now saves both the last checkpoint and the best checkpoint, instead of only the last checkpoint
- most operations loading models from HuggingFace can now receive an authentication token (useful to access private repositories)
- support for remapping entity labels in Seq2SeqEvaluator (useful when predicted and reference label do not match exactly)
- easier initialization of PASpeakerDetector

### Fixed

- medkit is now compatible with the latest (0.9) EDS-NLP
- custom attributes (DateAttribute, UMLSNormAttribute) don't have None as a value anymore
