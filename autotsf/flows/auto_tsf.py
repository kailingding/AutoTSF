# ==========================
# Author: Kailing Ding
# Created Date: 05/07/2020
# ==========================
import pandas as pd
import numpy as np

from flows.data_processing_and_transform import DataProcess
from flows.model_training import ModelTraining


class Autotsf:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self._data_processor = None
        self._model_trainer = None

    def run_pipeline(self, data, num_steps):
        # 1. data preprocess and transformation
        self._data_processor = DataProcess()
        X_scaled, y_scaled = self._data_processor.preprocess_data(data)

        # 2. model training
        model_trainer = ModelTraining()
        model_trainer.fit_select(X_scaled, y_scaled)

        # model selection
        self._model_trainer = model_trainer

    def forecast(self, num_step_forecast=1):
        # 3. forecast
        pred_days, y_preds = self._model_trainer.predict(self._data_processor,
                                                         num_steps=num_step_forecast)

        return pred_days, y_preds


if __name__ == '__main__':
    auto_ts = Autotsf()
    ca_0 = pd.read_csv("../sample_data/data/CA_0.csv", parse_dates=['datetime'])
    auto_ts.run_pipeline(ca_0, 1)
    pred_days, y_preds = auto_ts.forecast(num_step_forecast=7)
