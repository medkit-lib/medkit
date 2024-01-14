from __future__ import annotations

__all__ = [
    "MEDKIT_JSON_VERSION",
    "ContentType",
    "build_header",
    "check_header",
]

import enum
from typing import Any

MEDKIT_JSON_VERSION = "0.2"


class ContentType(enum.Enum):
    TEXT_DOCUMENT = "text_document"
    TEXT_DOCUMENT_LIST = "text_document_list"
    TEXT_ANNOTATION_LIST = "text_annotation_list"
    AUDIO_DOCUMENT = "audio_document"
    AUDIO_DOCUMENT_LIST = "audio_document_list"
    AUDIO_ANNOTATION_LIST = "audio_annotation_list"


def build_header(content_type: ContentType) -> dict[str, Any]:
    return {
        "version": MEDKIT_JSON_VERSION,
        "content_type": content_type.value,
    }


def check_header(data, expected_content_type: ContentType):
    # NB: when newer versions of the medkit json format are introduced,
    # migration functions will have to be implemented
    if data["version"] != MEDKIT_JSON_VERSION:
        msg = "Input file has incompatible medkit version"
        raise RuntimeError(msg)

    content_type = ContentType(data["content_type"])
    if content_type is not expected_content_type:
        msg = (
            f"Input file does not have expected {expected_content_type.value} content"
            f" type (has {content_type.value} instead)"
        )
        raise RuntimeError(msg)
