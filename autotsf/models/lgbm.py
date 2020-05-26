from .models import ModelInterface
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error, r2_score


def rmse(a, p):
    return np.sqrt(mean_squared_error(a, p))


DEFAULT_PATH = "lgbm/model.txt"


class LGBM_model(ModelInterface):

    DEFAULT_PARAMS = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"rmse", "l1", "l2"},
        "num_leaves": 25,
        "learning_rate": 0.005,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_split_gain": 0.5,
        "min_child_weight": 1,
        "min_child_samples": 5,
        "n_estimators": 5000,
    }

    def __init__(self, is_model_trained=False):
        self.is_model_trained = is_model_trained
        self.best_params = self.DEFAULT_PARAMS

    def tune_hyperparameters(self, X, y):
        params = {}
        self.best_params = params


    def train_model(self, X, y):
        if not self.is_model_trained:
            y = y.squeeze()

            idx = int(len(X) * 0.8)
            Xtr, Xval, ytr, yval = X[:idx], X[idx:], y[:idx], y[idx:]
            trainset = lgbm.Dataset(Xtr, ytr)
            evalset = lgbm.Dataset(Xval, yval, reference=trainset)

            params = self.best_params
            model = lgbm.train(
                params,
                trainset,
                num_boost_round=1000,
                valid_sets=evalset,
                verbose_eval=False,
                early_stopping_rounds=50,
            )
            self.model = model
            self.is_model_trained = True

    def evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        val_rmse = rmse(y_pred, y)
        val_r2 = r2_score(y_pred, y)
        return {"RMSE": val_rmse, "R2": val_r2}

    def save_model(self, path=DEFAULT_PATH):
        if self.model:
            self.model.save_model(
                path, num_iteration=self.model.best_iteration)

    def load_model(self, path=DEFAULT_PATH):
        self.model = lgbm.Booster(model_file=path)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    lgbm_model = LGBM_model()
    X = pd.read_csv("/Users/kailing/Desktop/model_x.csv").to_numpy()
    y = pd.read_csv("/Users/kailing/Desktop/model_y.csv").to_numpy()
    lgbm_model.train_model(X, y)
    print(lgbm_model.evaluate_model(X, y))
