import numpy as np
from autotsf.models.lgbm import LGBM_model
from autotsf.utils._testing import get_sample_data

def test_train_model():
    X = np.random.normal(0, 1,  (10, 3))
    y = np.random.normal(0, 1, (10, ))
    lgbm = LGBM_model()
    lgbm.train_model(X, y)
    assert lgbm.is_model_trained is True

def test_tune_hyperparameters():
    X = np.random.normal(0, 1, (10, 3))
    y = np.random.normal(0, 1, (10,))
    lgbm = LGBM_model()
    lgbm.tune_hyperparameters(X, y)
    assert lgbm.is_tuned is True


if __name__ == '__main__':
    # test_train_model()
    test_tune_hyperparameters()