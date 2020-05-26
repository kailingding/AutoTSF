# LSTM
import pandas as pd
from .models import ModelInterface
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
import numpy as np
from functools import partial
from keras.layers import Dense, Concatenate, Lambda, Input, Concatenate, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from hyperopt.pyll.base import scope
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# TODO:
# 1) fix batch_size
# 2) add space to instance variable field
# 3) Add status (is_trained, is_...)
# 4) log messages
# 5) Fix save & load
# 6) Try other loss for objective
# 7) K-fold


class LSTM_model(ModelInterface):
    """
    LSTM model
    """
    net_params = {
        "sequential": [13, 4, 12],
        "other": [9],
        "loss": "mean_squared_error",
    }

    tune_params = {
        "LSTM_kernal": 64,
        "dense_kernal": 64,
        "activation": "relu",
        "learning_rate": 1e-4,
        'dropout': 0.3,
        'batch_size': 64
    }

    train_params = {
        "train_epochs": 40,
    }

    tuning_params = {
        "tune_epoch": 40,
        "tune_validation": 0.1,
        "max_evals": 5,
        "folds": 10
    }

    MODEL_NOT_TRAIN_ERROR = "Model not trained. Please call train_model first."
    DEFAULT_PATH = "LSTM_MODEL/model.h5"

    def __init__(self):
        self.is_trained = False
        self.is_tuned = False

    def get_model(self, params):
        """
        get model
        """
        if self.is_tuned:
            print('best: {}'.format(params))
        num_of_variables = sum(params["sequential"]) + sum(params["other"])
        input = Input(shape=(1, num_of_variables,))
        outputs = []
        tmp = 0
        for i in params["sequential"]:
            data = Lambda(lambda x: x[:, :, tmp:tmp + i])(input)
            lstm = LSTM(int(params["LSTM_kernal"]), activation=params["activation"])(data)
            outputs.append(lstm)
            tmp += i
        for i in params["other"]:
            data = Lambda(lambda x: x[:, :, tmp:tmp + i])(input)
            dense = Dense(int(params["dense_kernal"]), activation=params["activation"])(Flatten()(data))
            outputs.append(dense)
            tmp += i
        concat = Concatenate()(outputs)
        dense1 = Dense(16, activation=params["activation"])(concat)
        dense2 = Dense(8, activation=params["activation"])(dense1)
        out = Dense(1, activation='tanh')(dense2)
        model = Model(inputs=input, outputs=out)
        # optimizer = net_params["optimizer"](learning_rate = net_params["learning_rate"])
        optimizer = Adam(lr=params["learning_rate"])
        model.compile(loss=params["loss"], optimizer=optimizer)
        return model

    def tune_hyperparameters(self, X, y):
        """
        Params:
        LSTM_kernal: numbers of kernal for LSTM layer
        dense_kernal: numbers of kernal for dense layers
        activation: activation function for all layers
        learning_rate: learning rate
        dropout: dropout rate for first dense layer and LSTM layer
        batch_size: batch size 
        """
        space = {
            'LSTM_kernal': scope.int(hp.quniform('LSTM_kernal', 16, 64, 16)),
            'dense_kernal': scope.int(hp.quniform('dense_kernal', 16, 64, 16)),
            'activation': hp.choice('activation', ['relu', 'tanh']),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.01)),
            'dropout': hp.uniform('dropout', 0.3, 0.7),
            # 'batch_size': hp.quniform('batch_size', 32, 512, 16)
        }
        ts = Trials()
        X = X.reshape(X.shape[0], 1, X.shape[1])
        best = fmin(lambda params: self.objective(params, X, y), space,
                    algo=partial(tpe.suggest, n_startup_jobs=15),
                    max_evals=self.tuning_params["max_evals"], trials=ts)
        self.best_param = best
        self.best_param['activation'] = ['relu', 'tanh'][self.best_param['activation']]
        print('best: {}'.format(self.best_param))

        self.is_tuned = True

    def objective(self, params, X, y):
        """
        Returns validation score from hyperparameters
        """
        params.update(self.net_params)
        model = self.get_model(params)
        hist = model.fit(X, y, epochs=self.tuning_params["tune_epoch"], verbose=0, validation_split=self.tuning_params["tune_validation"])
        loss = hist.history['val_loss']
        return {'loss': loss[-1], 'params': params, 'status': STATUS_OK}

    def save_model(self, path=DEFAULT_PATH):
        # if not self.is_model_trained:
        # 	raise Error(MODEL_NOT_TRAIN_ERROR)
        self.model.save(path)

    def load_model(self, path=DEFAULT_PATH):
        self.model = load_model(path)

    def evaluate_model(self, X, y):
        # if not self.is_model_trained:
        # 	raise Error()
        X = X.reshape(X.shape[0], 1, X.shape[1])
        predictions = self.model.predict(X)
        rmse = mean_squared_error(y, predictions)**0.5
        predictions = predictions.reshape(y.shape[0])
        y = y.reshape(y.shape[0])
        correlation_matrix = np.corrcoef(predictions, y)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy**2
        return {"RMSE": rmse, "R_squared": r_squared}

    def predict(self, X):
        # if not self.is_model_trained:
        # 	raise Error(MODEL_NOT_TRAIN_ERROR)
        print()
        X = X.reshape(X.shape[0], 1, X.shape[1])
        return self.model.predict(X)

    def train_model(self, X, y):
        if self.is_tuned == False:
            params = {}
            params.update(self.tune_params)
        else:
            params = self.best_param
        params.update(self.net_params)
        self.model = self.get_model(params)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        hist = self.model.fit(X, y, epochs=self.train_params["train_epochs"])
        self.is_trained = True
        return hist


if __name__ == "__main__":
    X = pd.read_csv("/Users/kailing/Desktop/model_x.csv").to_numpy()
    y = pd.read_csv("/Users/kailing/Desktop/model_y.csv").to_numpy()
    lstm = LSTM_model()
    lstm.tune_hyperparameters(X, y)
    lstm.train_model(X, y)
    # lstm.save_model()
    print(lstm.evaluate_model(X, y))
