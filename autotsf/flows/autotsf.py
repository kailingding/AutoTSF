# ==========================
# Author: Kailing Ding
# Created Date: 05/07/2020
# ==========================
import pandas as pd
import numpy as np

from .data_processing_and_transform import DataProcess
from .model_training import ModelTraining

# TODO:
#   1. Forecast return a pd dataframes
#   2. how to tests class and fucntions

class AutoTSF:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self._data_processor = None
        self._model_trainer = None

    def train(self, data):
        # 1. data preprocessing and transformation
        self._data_processor = DataProcess()
        X_scaled, y_scaled = self._data_processor.preprocess_data(data)

        # 2. model training
        model_trainer = ModelTraining()
        model_trainer.train_select(X_scaled, y_scaled)

        # model selection
        self._model_trainer = model_trainer

    def forecast(self, num_step_forecast=1):
        # 3. forecast
        pred_days, y_preds = self._model_trainer.predict(self._data_processor,
                                                         num_steps=num_step_forecast)
        y_preds = np.array(y_preds).flatten()

        pred_df = pd.DataFrame({"Datetime": pred_days, "Predict": y_preds})

        return pred_df
