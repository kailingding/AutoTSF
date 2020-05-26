import numpy as np
import pandas as pd
from datetime import timedelta

from data_preprocessing.calenderTrans import CalenderTransformer
from data_preprocessing.salesHistoryTrans import SalesHistoryTransformer

from sklearn.preprocessing import StandardScaler

'''
TODO:
    1. when creating new rows, use differenced signal or no
        because the forecasted  one ddoesn't have a y  (i.e. can't diff)
        (temp solu: use last day's y)
    2. loss of data, i.e. if lag = 365, first 365 will be lost
    3. Do we save raw_df as class attributes
'''


class DataProcess:
    '''
        1. Preprocess data integrated with following column transfomers
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
        '''
            Normalize signal
        '''

        # fit the normalizer with X
        self.scaler.fit(X)
        # scale train
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def inverse_scale(self, yhat_scaled):
        '''
        Inverse transform normalized_X back to X with original scale
        '''
        if isinstance(yhat_scaled, float):
            yhat_scaled = np.array([yhat_scaled])

        yhat = self.scaler.inverse_transform(yhat_scaled)

        return yhat

    def difference(self, arr, interval=1):
        '''
            Differencing arr
        '''
        diff = list()
        for i in range(interval, len(arr)):
            value = arr[i] - arr[i - interval]
            diff.append(value)
        # diff value at the last index
        diff.append(np.mean(diff))
        return diff

    def inverse_difference(self, yhat, arr, interval=1):
        return yhat + arr[-interval]

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
        df = pd.concat([df, sales_lag_df, calender_df], axis=1).iloc[self.max_lag:, :].copy()

        # split features and label
        df.drop(['datetime', 'signal', 'diff_signal'], axis=1, inplace=True)
        X_scaled = df.drop('diff_signal_scaled', axis=1).values
        y_scaled = df['diff_signal_scaled'].values

        return X_scaled, y_scaled

    def create_features(self, yhat_scaled=None):
        '''
            form a row for prediction
        '''
        # get next day
        day = np.datetime64(self.raw_df['datetime'].iloc[-1] + timedelta(days=1))

        last_row_ix = self.raw_df.index[-1]
        # TODO: 1
        if yhat_scaled:
            # inverse scale and inverse difference yhat
            # replace the temp one
            yhat_diff = self.inverse_scale(yhat_scaled)[0]
            yhat = self.inverse_difference(yhat_diff, self.raw_df['signal'].values)
            self.raw_df.loc[last_row_ix, 'signal'] = yhat
            self.raw_df.loc[last_row_ix, 'diff_signal'] = yhat_diff
            self.raw_df.loc[last_row_ix, 'diff_signal_scaled'] = yhat_scaled
        else:
            # create placeholder for signal on the first step forecast
            yhat = self.raw_df.loc[last_row_ix, 'signal']
            yhat_diff = self.raw_df.loc[last_row_ix, 'diff_signal']
            yhat_scaled = self.raw_df.loc[last_row_ix, 'diff_signal_scaled']

        # append new signal
        # TODO: DO we need to create a new df for new_row to append
        self.raw_df = self.raw_df.append({"datetime": day,
                                          "signal": yhat,
                                          "diff_signal": yhat_diff,
                                          "diff_signal_scaled": yhat_scaled},
                                         ignore_index=True)

        # create new features
        new_calender_df = CalenderTransformer().transform(
            self.raw_df[['datetime']])
        new_sales_lag_df, _ = SalesHistoryTransformer().transform(
            self.raw_df[['diff_signal_scaled']])
        new_row = pd.concat([new_sales_lag_df, new_calender_df],
                            axis=1).iloc[-1, :].copy()
        return new_row.values, yhat, day


if __name__ == "__main__":
    calenderTrans = DataProcess()
    ca_0 = pd.read_csv("../sample_data/data/CA_0.csv", parse_dates=['datetime'])
    calenderTrans.preprocess_data(ca_0)
    print(calenderTrans.create_features(0.2))
