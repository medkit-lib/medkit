import pytest
from numpy.testing import assert_almost_equal

from medkit.text.metrics.irr_utils import krippendorff_alpha


def test_krippendorff_alpha():
    # data from Krippendorff,K.(2011)
    # binary data, two annotators, no missing data
    annotator1 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    annotator2 = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

    alpha = krippendorff_alpha([annotator1, annotator2])
    assert_almost_equal(alpha, 0.095, decimal=3)
    assert alpha == krippendorff_alpha([annotator2, annotator1])

    # nominal data, two annotators, no missing data
    annotator1 = ["a", "a", "b", "b", "d", "c", "c", "c", "e", "d", "d", "a"]
    annotator2 = ["b", "a", "b", "b", "b", "c", "c", "c", "e", "d", "d", "d"]
    alpha = krippendorff_alpha([annotator1, annotator2])
    assert_almost_equal(alpha, 0.692, decimal=3)
    assert alpha == krippendorff_alpha([annotator2, annotator1])

    # nominal data, any number of annotators, missing data
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    alpha = krippendorff_alpha([A, B, C, D])
    assert_almost_equal(alpha, 0.743, decimal=3)
    assert alpha == krippendorff_alpha([D, C, B, A])
    assert alpha == krippendorff_alpha([D, A, C, B])

    # testing exceptions
    with pytest.raises(
        AssertionError, match="Number of labels should be the same for all .*"
    ):
        krippendorff_alpha([[1, 2, 1], [1, 2, 1] * 2])

    with pytest.raises(AssertionError, match="There must be more than one .*"):
        krippendorff_alpha([[1, 1, 1], [1, 1, 1]])
