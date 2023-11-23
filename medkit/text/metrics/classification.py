"""
This module needs extra-dependencies not installed as core dependencies of medkit.
To install them, use `pip install medkit-lib[metrics-text-classification]`.
"""

__all__ = ["TextClassificationEvaluator"]
import logging
from typing import Dict, List, Union
from typing_extensions import Literal

from sklearn.metrics import classification_report, cohen_kappa_score

from medkit.core.text import TextDocument
from medkit.text.metrics.irr_utils import krippendorff_alpha

logger = logging.getLogger(__name__)


class TextClassificationEvaluator:
    """An evaluator for attributes of TextDocuments"""

    def __init__(self, attr_label: str):
        """Initialize the text classification evaluator

        Parameters
        ----------
        attr_label:
            Label of the attribute to evaluate.
        """
        self.attr_label = attr_label

    def _extract_attr_values(
        self, docs: List[TextDocument]
    ) -> List[Union[str, int, bool]]:
        """Prepare docs attrs to compute the metric

        Parameters
        ----------
        docs : List[TextDocument]
            List of documents with attributes

        Returns
        -------
        attr_values :  List[Union[str,int,bool]]
            List with the representation of the attribute by document.
        """
        attr_values = []
        for doc in docs:
            attrs = doc.attrs.get(label=self.attr_label)

            if not attrs:
                raise ValueError(
                    f"No attribute with label {self.attr_label} was found in the"
                    " document"
                )
            if len(attrs) > 1:
                logger.warning(
                    f"Found several attributes with label '{self.attr_label}', ignoring"
                    " all but first"
                )

            attr_value = attrs[0].value
            if not isinstance(attr_value, (str, int, bool)):
                raise ValueError(
                    "The type of the attr value is not supported by this evaluator."
                    "Only str,int or bool are supported."
                )

            attr_values.append(attr_value)
        return attr_values

    def compute_classification_report(
        self,
        true_docs: List[TextDocument],
        predicted_docs: List[TextDocument],
        metrics_by_attr_value: bool = True,
        average: Literal["macro", "weighted"] = "macro",
    ) -> Dict[str, Union[float, int]]:
        """Compute classification metrics of document attributes giving annotated documents.
        This method uses `sklearn.metrics.classification_report` to compute
        precision, recall and F1-score for value of the attribute.

        .. warning::
            The set of true and predicted documents must be sorted to calculate the metric

        Parameters
        ----------
        true_docs:
            Text documents containing attributes of reference
        predicted_docs:
            Text documents containing predicted attributes
        metrics_by_attr_value:
            Whether return metrics by attribute value.
            If False, only global metrics are returned
        average:
            Type of average to be performed in metrics.
            - `macro`, unweighted mean (default)
            - `weighted`, weighted average by support (number of true instances by attr value)

        Returns
        -------
        Dict[str,Union[float,int]]:
            A dictionary with the computed metrics
        """
        true_tags = self._extract_attr_values(true_docs)
        pred_tags = self._extract_attr_values(predicted_docs)

        report = classification_report(
            y_true=true_tags,
            y_pred=pred_tags,
            output_dict=True,
            zero_division=0,
        )

        scores = {
            f"{average}_{key}": value for key, value in report[f"{average} avg"].items()
        }
        scores["support"] = scores.pop(f"{average}_support")
        scores["accuracy"] = report.pop("accuracy")

        if metrics_by_attr_value:
            for value_key in report:
                if value_key.endswith("avg"):
                    continue

                for metric_key, metric_value in report[value_key].items():
                    scores[f"{value_key}_{metric_key}"] = metric_value

        return scores

    def compute_cohen_kappa(
        self, docs_annotator_1: List[TextDocument], docs_annotator_2: List[TextDocument]
    ) -> Dict[str, Union[float, int]]:
        """Compute the cohen's kappa score, an inter-rated agreement score between two annotators.
        This method uses 'sklearn' as backend to compute the level of agreement.

        .. warning::
            The set of documents must be sorted to calculate the metric

        Parameters
        ----------
        docs_annotator_1:
            Text documents containing attributes annotated by the first annotator
        docs_annotator_2:
            Text documents to compare, these documents contain attributes
            annotated by the other annotator

        Returns
        -------
        Dict[str, Union[float, int]]:
            A dictionary with cohen's kappa score and support (number of annotated docs).
            The value is a number between -1 and 1, where 1 indicates perfect agreement; zero
            or lower indicates chance agreement.
        """
        ann1_tags = self._extract_attr_values(docs_annotator_1)
        ann2_tags = self._extract_attr_values(docs_annotator_2)

        scores = {
            "cohen_kappa": cohen_kappa_score(y1=ann1_tags, y2=ann2_tags),
            "support": len(ann1_tags),
        }

        return scores

    def compute_krippendorff_alpha(
        self, docs_annotators: List[List[TextDocument]]
    ) -> Dict[str, Union[float, int]]:
        """Compute the Krippendorff alpha score, an inter-rated agreement score between
        multiple annotators.

        .. warning::
            Documents must be sorted to calculate the metric.

        .. note::
            See :mod:`medkit.text.metrics.irr_utils.krippendorff_alpha` for more information about the score

        Parameters
        ----------
        docs_annotators:
            A list of list of Text documents containing attributes.
            The size of the list is the number of annotators to compare.

        Returns
        -------
        Dict[str, Union[float,int]]:
            A dictionary with the krippendorff alpha score, number of annotators and support (number of documents).
            A value of 1 indicates perfect reliability between annotators; zero or lower indicates
            absence of reliability.
        """
        if len(docs_annotators) < 2 or not isinstance(docs_annotators[0], list):
            raise ValueError(
                "'docs_annotators' should contain at least two list of TextDocuments to"
                " compare"
            )

        all_annotators_data = []

        for docs in docs_annotators:
            annotator_tags = self._extract_attr_values(docs)
            all_annotators_data.append(annotator_tags)
        scores = {
            "krippendorff_alpha": krippendorff_alpha(all_annotators_data),
            "nb_annotators": len(all_annotators_data),
            "support": len(all_annotators_data[0]),
        }

        return scores
