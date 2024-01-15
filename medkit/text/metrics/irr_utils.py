"""Metrics to assess inter-annotator agreement"""
from __future__ import annotations

__all__ = ["krippendorff_alpha"]

import numpy as np


def _get_values_by_unit_matrix(reliability_data: np.ndarray, labels_set: np.ndarray) -> np.ndarray:
    """Return the label counts given the annotators_data.

    Parameters
    ----------
    reliability_data : ndarray
        numpy array with labels given to `n_samples` by `m_annotators`
        The missing labels are represented with `None`.

    labels_set : ndarray
        Possible labels the item can take.

    Returns
    -------
    ndarray
        Number of annotators that assigned a certain label by annotation.
        Where `n_labels` is the number of possible labels and `n_samples`
        is the number of annotations.
    """
    ann_data_expanded = np.expand_dims(reliability_data, 2)
    return np.sum(ann_data_expanded == labels_set, axis=0).T


def _compute_observed_disagreement(values_by_unit_matrix: np.ndarray) -> float:
    """Return the observed disagreement given values-by-unit matrix.

    Parameters
    ----------
    values_by_unit_matrix : ndarray
        Count of annotators that assigned a certain label by annotation.

    Returns
    -------
    float
        observed disagreement among labels assigned to annotations
    """
    # select only units with disagreement
    # units with more than two assigned values
    units_to_keep = np.count_nonzero(values_by_unit_matrix, 0) > 1
    matrix_disagreement = values_by_unit_matrix[:, units_to_keep]
    total_by_unit = matrix_disagreement.sum(0)

    do = 0
    for u, unit in enumerate(matrix_disagreement.T):
        positive_unit = unit[unit > 0]
        for n in range(len(positive_unit)):
            # only nominal weight is supported in this function
            # perfect agreement seen as 0 disagreement
            p_unit = np.dot(positive_unit[n], positive_unit[n + 1 :]) / (total_by_unit[u] - 1)
            do += np.sum(p_unit)
    return do


def _compute_expected_disagreement(values_by_unit_matrix: np.ndarray) -> float:
    """Return the expected disagreement given values-by-unit matrix.

    Parameters
    ----------
    values_by_unit_matrix : ndarray
        Count of annotators that assigned a certain label by annotation.

    Returns
    -------
    float
        expected disagreement annotators will have by chance
    """
    # all units with at least 1 value
    paried_units = values_by_unit_matrix.sum(0) > 1
    total_by_value = values_by_unit_matrix[:, paried_units].sum(1)

    de = 0
    # only nominal weight is supported in this function
    for n_c in range(len(total_by_value) - 1):
        de += np.sum(np.dot(total_by_value[n_c], total_by_value[n_c + 1 :]))
    return de


def krippendorff_alpha(all_annotators_data: list[list[None | str | int]]) -> float:
    """Compute Krippendorff's alpha: a coefficient of agreement among many
    annotators.

    This coefficient is a generalization of several reliability indices.
    The general form is:

    .. math::
        \\alpha = 1 - \\frac{D_o}{D_e}

    where :math:`D_o` is the observed disagreement among labels assigned to
    units or annotations and :math:`D_e` is the disagreement between annotators
    attributable to chance. The arguments of the disagreement measures are values
    in coincidence matrices.

    This function implements the general computational form proposed in [1]_,
    but only supports binaire or nominal labels.

    Parameters
    ----------
    all_annotators_data : list of list of str or int or None
        Reliability_data, list or array of labels given to `n_samples` by `m_annotators`.
        Missing labels are represented with `None`

    Returns
    -------
    float
        The alpha coefficient, a number between 0 and 1.
        A value of 0 indicates the absence of reliability, and
        a value of 1 indicates perfect reliability.

    Raises
    ------
    AssertionError
        Raise if any list of labels within `all_annotators_data` differs in size or
        if there is only one label to be compared.

    References
    ----------
    .. [1] K. Krippendorff, “Computing Krippendorff's alpha-reliability,”
            ScholarlyCommons, 25-Jan-2011, pp. 8-10. [Online].
            Available: https://repository.upenn.edu/asc_papers/43/

    Examples
    --------
    Three annotators labelled six items. Some labels are missing.

    >>> annotator_A = ["yes", "yes", "no", "no", "yes", None]
    >>> annotator_B = [None, "yes", "no", "yes", "yes", "no"]
    >>> annotator_C = ["yes", "no", "no", "yes", "yes", None]
    >>> krippendorff_alpha([annotator_A, annotator_B, annotator_C])
    0.42222222222222217
    """
    assert all(
        len(d) == len(all_annotators_data[0]) for d in all_annotators_data
    ), "Number of labels should be the same for all annotators"

    all_annotators_data = np.asarray(all_annotators_data)
    labels_set = np.asarray(list({x for x in all_annotators_data.flatten() if x is not None}))
    assert len(labels_set) > 1, "There must be more than one label in annotators data"

    values_count = _get_values_by_unit_matrix(all_annotators_data, labels_set)
    do = _compute_observed_disagreement(values_count)
    de = _compute_expected_disagreement(values_count)
    total_paried_values = np.sum(values_count[:, values_count.sum(0) > 1])

    return 1 - (total_paried_values - 1) * (do / de)
