from __future__ import annotations

__all__ = ["DucklingMatcher"]

import warnings
from typing import Iterator

import requests

from medkit.core import Attribute
from medkit.core.text import Entity, NEROperation, Segment, span_utils


class DucklingMatcher(NEROperation):
    """Entity annotator using Duckling (https://github.com/facebook/duckling).

    This annotator can parse several types of information in multiple languages:
        amount of money, credit card numbers, distance, duration, email, numeral,
        ordinal, phone number, quantity, temperature, time, url, volume.

    This annotator currently requires a Duckling Server running. The easiest method is
    to run a docker container :

    >>> docker run --rm -d -p <PORT>:8000 --name duckling rasa/duckling:<TAG>

    This command will start a Duckling server listening on port <PORT>.
    The version of the server is identified by <TAG>
    """

    def __init__(
        self,
        output_label: str,
        version: str,
        url: str = "http://localhost:8000",
        locale: str = "fr_FR",
        dims: list[str] | None = None,
        attrs_to_copy: list[str] | None = None,
        uid: str | None = None,
    ):
        """Instantiate the Duckling matcher

        Parameters
        ----------
        output_label : str
            Label to use for attributes created by this annotator.
        version : str
            Version of the Duckling server.
        url : str, optional
            URL of the server. Defaults to "http://localhost:8000"
        locale : str, default="fr_FR"
            Language flag of the text to parse following ISO-639-1 standard, e.g. "fr_FR"
        dims : list of str, optional
            List of dimensions to extract. If None, all available dimensions will be extracted.
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the source segment
            to the created entity. Useful for propagating context attributes
            (negation, antecendent, etc)
        uid : str, optional
            Identifier of the matcher
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.output_label: str = output_label
        self.version: str = version
        self.url: str = url
        self.locale: str = locale
        self.dims: list[str] | None = dims
        self.attrs_to_copy: list[str] = attrs_to_copy

        self._test_connection()

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Return entities for each match in `segments`

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for matches

        Returns
        -------
        list of Entity
            Entities found in `segments`
        """
        return [entity for segment in segments for entity in self._find_matches_in_segment(segment)]

    def _find_matches_in_segment(self, segment: Segment) -> Iterator[Entity]:
        payload = {
            "locale": self.locale,
            "text": segment.text,
        }
        if self.dims is not None:
            # manually encode dim strings because we need to be like
            # 'dims=["time", "duration"]' but requests will encode it to 'dims=time&dims=duration'
            # also note that we must use double quotes, not single quotes
            payload["dims"] = str(self.dims).replace("'", '"')
        api_result = requests.post(f"{self.url}/parse", data=payload, timeout=10)
        api_result.raise_for_status()

        matches = api_result.json()
        for match in matches:
            if self.dims is not None and match["dim"] not in self.dims:
                warnings.warn("Dims are not properly filtered by duckling API call", stacklevel=2)
                continue

            text, spans = span_utils.extract(segment.text, segment.spans, [(match["start"], match["end"])])

            entity = Entity(
                label=match["dim"],
                text=text,
                spans=spans,
            )

            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    entity.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            norm_attr = Attribute(
                label=self.output_label,
                value=match["value"],
                metadata={
                    "version": self.version,
                },
            )
            entity.attrs.add(norm_attr)

            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])
                self._prov_tracer.add_prov(norm_attr, self.description, source_data_items=[segment])

            yield entity

    def _test_connection(self):
        api_result = requests.get(self.url, timeout=10)
        api_result.raise_for_status()
