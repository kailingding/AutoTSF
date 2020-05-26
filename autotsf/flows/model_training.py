# ==========================
# Author: Kailing Ding
# Created Date: 05/06/2020
# ==========================

import pandas as pd
import numpy as np

from models.lstm import LSTM_model
from models.lgbm import LGBM_model

# TODO:
#   1. Return inverse scaled differenced yhat

class ModelTraining:
    '''
        class for model training and model selection
    '''

    def __init__(self):
        self.best_model_ = None
        self._is_model_tuned = False

    def fit_select(self, X, y, model=None):
        lstm = LSTM_model()

        # training phase
        lstm.train_model(X, y)

        # tuning phase
        if not self._is_model_tuned:
            lstm.tune_hyperparameters(X, y)
            self._is_model_tuned = True

        # TODO: select best model
        self.best_model_ = lstm

    def _ind_predict(self, new_row):
        # predict one-step
        y_pred = self.best_model_.predict(new_row)
        return y_pred

    def predict(self, data_processor, num_steps=1, pass_model_file=None):
        y_preds = []
        pred_days = []    
        for i in range(num_steps):
            if i == 0:
                # prepare for first step forecasting
                new_row, yhat, day = data_processor.create_features()
            else:
                # prepare for multi-step forecasting
                new_row, yhat, day = data_processor.create_features(y_preds[-1])

            if new_row.ndim == 1:
                new_row = new_row[np.newaxis, :]
            y_pred = self._ind_predict(new_row)[0]
            y_preds.append(y_pred)
            pred_days.append(day)

        return pred_days, y_preds 

if __name__ == '__main__':
    model_trainer = ModelTraining()
    X = pd.read_csv("/Users/kailing/Desktop/model_x.csv").to_numpy()
    y = pd.read_csv("/Users/kailing/Desktop/model_y.csv").to_numpy()
    model_trainer.fit_select(X, y)

