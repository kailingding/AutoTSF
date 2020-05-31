import pandas as pd
import numpy as np
from autotsf.flows.data_processing_and_transform import DataProcess


def test_scale():
    data_processer = DataProcess()
    X = pd.DataFrame({0: [-1, 0, 1]})
    X_scaled = data_processer.scale(X)
    assert np.round(X_scaled[0][0], 2) == -1.22


def test_inverse_scale():
    data_processer = DataProcess()
    X = pd.DataFrame({0: [1., 2., 3., 4.]})
    X_scaled = data_processer.scale(X)
    X_temp = data_processer.inverse_scale(X_scaled)
    np.testing.assert_array_equal(X[0].values, X_temp)


def test_difference():
    data_processer = DataProcess()
    arr = [1, 2, 3, 4]
    diff_arr = data_processer.difference(arr, interval=1)
    np.testing.assert_array_equal(diff_arr, [1., 1., 1., 1.])


def test_inverse_difference():
    data_processer = DataProcess()
    arr = [1, 3, 4, 10]
    yhat = 3
    inversed = data_processer.inverse_difference(yhat, arr)
    assert inversed == 13


def test_preprocess_data():
    pass


def test_create_features():
    pass


if __name__ == "__main__":
    # test_scale()
    # test_inverse_scale()
    test_difference()
