"""This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[nlstruct]`.
"""
from __future__ import annotations

__all__ = ["NLStructEntityMatcher"]

from pathlib import Path
from typing import Iterator

import huggingface_hub
import nlstruct
import torch

from medkit.core import Attribute
from medkit.core.text import Entity, NEROperation, Segment, span_utils

_PYTORCH_FILES = ["*.ckpt", "*.pt"]
_TXT_FILES = "*.txt"

# name of nlstruct registry to fix the config
_CONCAT_REGISTRY_NAME = "concat"
_EMBEDDING_REGISTRY_NAME = "word_embeddings"


class NLStructEntityMatcher(NEROperation):
    """Entity matcher based on a NLstruct InformationExtraction model.
    The matcher expects a directory with a torch checkpoint and a text file if
    the model was pretrained using word embeddings.

    The paper [1]_ presents a model trained with the NLstruct [2]_ library and the mimic learning approach.
    The model used a private teacher model to annotate the unlabeled [CAS clinical French corpus](https://aclanthology.org/W18-5614/).
    The weights of the CAS student model are shared via the HuggingFace Hub and you can
    load the model using the following model name `NesrineBannour/CAS-privacy-preserving-model`
    to create a NLstructEntityMatcher.

    References
    ----------
    .. [1] Nesrine Bannour, Perceval Wajsbürt, Bastien Rance, Xavier Tannier, and Aurélie Névéol. 2022.
            Privacy-preserving mimic models for clinical named entity recognition in French.
            Journal of Biomedical Informatics 130, (2022), 104073.
            DOI: https://doi.org/https://doi.org/10.1016/j.jbi.2022.104073
    .. [2] Perceval Wajsbürt. 2021. Extraction and normalization of simple and structured entities in medical documents.
            Theses. Sorbonne Université. Retrieved from https://hal.archives-ouvertes.fr/tel-03624928

    """

    def __init__(
        self,
        model_name_or_dirpath: str | Path,
        attrs_to_copy: list[str] | None = None,
        device: int = -1,
        hf_auth_token: str | None = None,
        cache_dir: str | Path | None = None,
        name: str | None = None,
        uid: str | None = None,
    ):
        """Parameters
        ----------
        model_name_or_dirpath : str or Path
            Name (on the HuggingFace models hub) or dirpath of the NLstruct model.
            The model dir must contain a PyTorch file ('.cpkt','.pt') and a text file (.txt)
            representing the FastText embeddings if required.
        attrs_to_copy : list of str, optional
            Labels of the attributes that should be copied from the input segment
            to the created entity. Useful for propagating context attributes
            (negation, antecendent, etc).
        device : int, default=-1
            Device to use for the NLstruct model. Follows the HuggingFace convention
            (-1 for "cpu" and device number for gpu, for instance 0 for "cuda:0").
        hf_auth_token : str, optional
            HuggingFace Authentication token (to access private models on the
            hub)
        cache_dir : str or Path, optional
            Directory where to store downloaded models. If not set, the default
            HuggingFace cache dir is used.
        name : str, optional
            Name describing the matcher (defaults to the class name).
        uid : str, optional
            Identifier of the matcher.
        """
        # Pass all arguments to super (remove self and confidential hf_auth_token)
        init_args = locals()
        init_args.pop("self")
        init_args.pop("hf_auth_token")
        super().__init__(**init_args)

        if attrs_to_copy is None:
            attrs_to_copy = []

        self.cache_dir = cache_dir
        self.attrs_to_copy = attrs_to_copy
        self.model_name_or_dirpath = Path(model_name_or_dirpath)

        # get checkpoint dir
        if self.model_name_or_dirpath.exists():
            checkpoint_dir = self.model_name_or_dirpath
        else:
            allow_patterns = [*_PYTORCH_FILES, _TXT_FILES]
            # download only allowed files
            checkpoint_dir = huggingface_hub.snapshot_download(
                repo_id=str(model_name_or_dirpath),
                cache_dir=self.cache_dir,
                allow_patterns=allow_patterns,
                token=hf_auth_token,
            )
            checkpoint_dir = Path(checkpoint_dir)

        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.model = self._load_from_checkpoint_dir(checkpoint_dir, self.device)
        self.model.eval()

    @staticmethod
    def _load_from_checkpoint_dir(checkpoint_dir: Path, device):
        """Get the location of the checkpoint and fix the path of the Fast Text file
        in the configuration. Return the nlstruct model created with the modified config.
        """
        checkpoint_filepaths = [filepath for pattern in _PYTORCH_FILES for filepath in checkpoint_dir.glob(pattern)]

        if not len(checkpoint_filepaths):
            msg = f"There was no PyTorch file with a NLstruct checkpoint in '{checkpoint_dir.name}'"
            raise FileNotFoundError(msg)

        # BUGFIX: (nlstruct) The config created from nlstruct defines a filename
        # without a relative path. This means that the text file needs to be in
        # the same place where the object is created.
        # Cf. 'nlstruct.common.WordEmbeddings'

        # Force the filename to use the checkpoint directory
        checkpoint_filepath = checkpoint_filepaths[0]
        loaded = torch.load(checkpoint_filepath, map_location=device)
        config = loaded["config"]

        # modify config if the encoder is a 'concat'model
        if config["encoder"]["module"] == _CONCAT_REGISTRY_NAME:
            # looks for the 'word_embeddings' config
            key_and_filename = [
                (key, data["filename"])
                for key, data in config["encoder"]["encoders"].items()
                if data["module"] == _EMBEDDING_REGISTRY_NAME
            ]

            if len(key_and_filename) != 0:
                key, filename = key_and_filename[0]
                # if 'filename' is empty, pretrained without embeddings (c.f nlstruct)
                # keep the same config
                if filename:
                    new_path = checkpoint_dir / Path(filename).name

                    if not new_path.exists():
                        msg = f"The text file '{new_path}' with the fast text embeddings does not exist"
                        raise ValueError(msg)

                    # update the filename of the wordEmbeddings model
                    config["encoder"]["encoders"][key]["filename"] = str(new_path)

        # similar to nlstruct load pretrained
        # create the model using modified config
        model = nlstruct.get_instance(config)
        model.load_state_dict(loaded["state_dict"], strict=False)
        return model

    def run(self, segments: list[Segment]) -> list[Entity]:
        """Return entities for each match in `segments`.

        Parameters
        ----------
        segments : list of Segment
            List of segments into which to look for matches.

        Returns
        -------
        list of Entity
            Entities found in `segments`.
        """
        # predict matches by segments
        entities = []
        for segment in segments:
            matches = self.model.predict({"doc_id": segment.uid, "text": segment.text})
            entities.extend(self._matches_to_entities(matches, segment))
        return entities

    def _matches_to_entities(self, matches: list[dict], segment: Segment) -> Iterator[Entity]:
        for match in matches["entities"]:
            text_all, spans_all = [], []

            # build entity by fragments
            for fragment in match["fragments"]:
                text, spans = span_utils.extract(segment.text, segment.spans, [(fragment["begin"], fragment["end"])])
                text_all.append(text)
                spans_all.extend(spans)

            text_all = "".join(text_all)

            # support multilabel
            label = match["label"] if isinstance(match["label"], str) else "-".join(match["label"])
            entity = Entity(
                label=label,
                text=text_all,
                spans=spans_all,
            )

            # TBD: This confidence is not well described,
            # normally around 0.99, round to avoid problems in export
            score_attr = Attribute(label="confidence", value=float("{:.2f}".format(match["confidence"])))
            entity.attrs.add(score_attr)

            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(entity, self.description, source_data_items=[segment])
                self._prov_tracer.add_prov(score_attr, self.description, source_data_items=[segment])

            # copy attrs from segment
            for label in self.attrs_to_copy:
                for attr in segment.attrs.get(label=label):
                    copied_attr = attr.copy()
                    entity.attrs.add(copied_attr)
                    # handle provenance
                    if self._prov_tracer is not None:
                        self._prov_tracer.add_prov(copied_attr, self.description, [attr])

            yield entity
