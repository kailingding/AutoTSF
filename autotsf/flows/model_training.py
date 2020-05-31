# ==========================
# Author: Kailing Ding
# Created Date: 05/06/2020
# ==========================

import pandas as pd
import numpy as np

from ..models.lstm import LSTM_model
from ..models.lgbm import LGBM_model
from sklearn.linear_model import LinearRegression, Ridge

# TODO:
#   1. Return inverse scaled differenced yhat
#   2. More time-series models

class ModelTraining:
    '''
        class for model training and model selection
    '''

    def __init__(self):
        self.models = []
        self.best_model_ = None
        self._is_model_tuned = False

    def train_select(self, X, y, model=None):
        # initialize model
        lstm = LSTM_model()
        lgbm = LGBM_model()
        self.models.extend([lstm, lgbm])

        # training phase
        lstm.train_model(X, y)
        lgbm.train_model(X, y)

        # tuning phase
        if not self._is_model_tuned:
            lstm.tune_hyperparameters(X, y)
            lgbm.tune_hyperparameters(X, y)
            self._is_model_tuned = True

        best_model, best_score = None, -float('inf')
        for model in self.models:
            score = model.evaluate_model(X, y)['R2']
            if score > best_score:
                self.best_model_ = model
        # self.best_model_ = Ridge()
        # self.best_model_.fit(X, y)

    def _ind_predict(self, new_row):
        # predict one-step
        y_pred = self.best_model_.predict(new_row)
        return y_pred

    def predict(self, data_processor, num_steps=1):
        y_preds = []
        y_preds_diff_scaled = []
        pred_days = []    
        for i in range(num_steps):
            if i == 0:
                # prepare for first step forecasting
                new_row, day = data_processor.create_features()
            else:
                # prepare for multi-step forecasting
                new_row, day = data_processor.create_features(y_preds_diff_scaled[-1])

            if new_row.ndim == 1:
                new_row = new_row[np.newaxis, :]
            y_pred_diff_scaled = self._ind_predict(new_row)[0]
            # invert y_pred
            y_pred_diff = data_processor.inverse_scale(y_pred_diff_scaled)
            y_pred = data_processor.inverse_difference(y_pred_diff,
                                                       data_processor.raw_df.signal.values)
            # append to the y preds list
            y_preds.append(y_pred)
            y_preds_diff_scaled.append(y_pred_diff_scaled)
            pred_days.append(day)

        # reset processor
        data_processor.reset(num_steps)

        return pred_days, y_preds 


if __name__ == '__main__':
    model_trainer = ModelTraining()
    X = pd.read_csv("/Users/kailing/Desktop/model_x.csv").to_numpy()
    y = pd.read_csv("/Users/kailing/Desktop/model_y.csv").to_numpy()
    model_trainer.fit_select(X, y)
