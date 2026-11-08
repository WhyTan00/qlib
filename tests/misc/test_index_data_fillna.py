# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest

from qlib.utils.index_data import MultiData, SingleData


@pytest.mark.parametrize("data_class", [SingleData, MultiData])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("fill_value", [0.0, 7.0, np.inf, -np.inf, np.nan])
def test_fillna_preserves_nonmissing_values(data_class, inplace, fill_value):
    values = np.array([np.nan, np.inf, -np.inf, 3.0])
    expected = values.copy()
    expected[0] = fill_value
    if data_class is SingleData:
        data = data_class(values, index=["a", "b", "c", "d"])
    else:
        values = values.reshape(2, 2)
        expected = expected.reshape(2, 2)
        data = data_class(values, index=["a", "b"], columns=["x", "y"])
    original = data.data.copy()
    indices = data.indices.copy()

    result = data.fillna(fill_value, inplace=inplace)

    if inplace:
        assert result is None
        result = data
    else:
        assert result is not data
        np.testing.assert_array_equal(data.data, original)
        assert not np.shares_memory(result.data, data.data)
    assert result.indices == indices
    np.testing.assert_array_equal(result.data, expected)


def test_add_preserves_infinity_when_filling_missing_values():
    left = SingleData([np.inf, np.nan], index=["a", "b"])
    right = SingleData([1.0, 2.0], index=["a", "b"])

    result = left.add(right)

    np.testing.assert_array_equal(result.data, [np.inf, 2.0])
