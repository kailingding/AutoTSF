import numpy as np
import pandas as pd
from datetime import timedelta

from ..data_preprocessing.calenderTrans import CalenderTransformer
from ..data_preprocessing.salesHistoryTrans import SalesHistoryTransformer

from sklearn.preprocessing import StandardScaler

'''
TODO:
    1. loss of data, i.e. if lag = 365, first 365 will be lost
    2. Do we save raw_df as class attributes
'''


class DataProcess:
    '''
        1. Preprocess data integrated with following column transformers
                a. SalesHistoryTransformer
                b. CalenderTransformer
                c. normalize data
                d. generate X_pred for prediction
    '''

    def __init__(self, is_data_processed=False):
        self.scaler = StandardScaler()
        self.raw_df = None
        self.max_lag = None

    def scale(self, X):
        # fit the normalizer with X
        self.scaler.fit(X)
        # scale train
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def inverse_scale(self, yhat_scaled):
        # Inverse transform normalized_X back to X with original scale
        if isinstance(yhat_scaled, float):
            yhat_scaled = np.array([yhat_scaled])

        yhat = self.scaler.inverse_transform(yhat_scaled)
        
        # flatten yhat
        yhat = np.array(yhat).flatten()

        # return a single float
        return yhat[0]

    def difference(self, arr, interval=1):
        diff = list()
        for i in range(interval, len(arr)):
            value = arr[i] - arr[i - interval]
            diff.append(value)
        # diff value at the last index
        diff = [np.mean(diff)] + diff
        return diff

    def inverse_difference(self, yhat, arr, interval=1):
        # return float
        return yhat + arr[-interval-1]

    def preprocess_data(self, df):
        '''
        Preprocess data and trasnform data
                Steps:
                        1. impute all null values with 0
                        2. adding more features such as lag and calender to the training set
                        3. standardize data
        '''

        # check if ['dateime', 'value'] is in X.columns
        if np.isin(['datetime', 'signal'], df.columns).sum() != 2:
            raise Exception('[datetime, signal] are not in X columns')

        # fill nan with 0 in signal column
        df['signal'] = df['signal'].fillna(0)

        # transform data to be stationary and differenced
        df['diff_signal'] = self.difference(df['signal'])
        df['diff_signal_scaled'] = self.scale(df[['diff_signal']])

        # save raw df
        self.raw_df = df.copy()

        # timeseries to supervised data
        sales_lag_df, self.max_lag = SalesHistoryTransformer().transform(df[['diff_signal_scaled']])
        calender_df = CalenderTransformer().transform(df[['datetime']])

        # TODO: 2
        X_scaled = pd.concat([sales_lag_df, calender_df], axis=1).iloc[self.max_lag:, :].values
        y_scaled = df['diff_signal_scaled'].iloc[self.max_lag:].values

        # split the last row with the rest
        # because last row is for forecasting
        X_scaled = X_scaled[:-1, :]
        self.X_scaled_forecast = X_scaled[-1, :]
        y_scaled = y_scaled[1:]

        return X_scaled, y_scaled

    def create_features(self, yhat_scaled=None):
        '''
            form a row for prediction
        '''
        # get next day
        day = np.datetime64(self.raw_df['datetime'].iloc[-1] + timedelta(days=1))

        last_row_ix = self.raw_df.index[-1]
        if yhat_scaled:
            # inverse scale and inverse difference yhat
            # replace the temp one
            yhat_diff = self.inverse_scale(yhat_scaled)
            yhat = self.inverse_difference(yhat_diff, self.raw_df['signal'].values)
            self.raw_df.loc[last_row_ix, 'signal'] = yhat
            self.raw_df.loc[last_row_ix, 'diff_signal'] = yhat_diff
            self.raw_df.loc[last_row_ix, 'diff_signal_scaled'] = yhat_scaled

            # create new features
            new_calender_df = CalenderTransformer().transform(
                self.raw_df[['datetime']])
            new_sales_lag_df, _ = SalesHistoryTransformer().transform(
                self.raw_df[['diff_signal_scaled']])
            new_row = pd.concat([new_sales_lag_df, new_calender_df],
                                axis=1).iloc[-1, :].copy()
            new_row = new_row.values
        else:
            new_row = self.X_scaled_forecast

        # create a new row for prediction
        self.raw_df = self.raw_df.append({"datetime": day}, ignore_index=True)
        return new_row, day

    def reset(self, num_step):
        self.raw_df = self.raw_df.iloc[:-num_step, :]


if __name__ == "__main__":
    calenderTrans = DataProcess()
    ca_0 = pd.read_csv("../sample_data/data/sample_data_1.csv", parse_dates=['datetime'])
    calenderTrans.preprocess_data(ca_0)
    print(calenderTrans.create_features(0.2))
