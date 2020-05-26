# ==========================
# Author: Kailing Ding
# Created Date: 05/05/2020
# ==========================

import numpy as np


class SalesHistoryTransformer:
    '''
        A column transformer that add more sales-related features to X
    '''

    def __init__(self):
        # if size of X * 0.3 > rolling_size, then create that new feature
        self.threshold_for_rolling = 0.3
        self.lags = list(np.arange(1, 15))
        self.lags.extend([28, 182, 364])
        self.roll_groups = [7, 14, 28, 364]

    def transform(self, X, y=None):
        '''
                Roll value based on various window size

                Elements off the end of the array will be droppeds. 

        Parameters
        ----------
        X : array_like
            Input array

        Features Engineering
        --------------------
        List of variable added to the training set:
                1. 1_days_ago (float): The amount sold 1 day before the date of sale
                2. 2_days_ago (float): The amount sold 2 days before the date of sale
                3. 3_days_ago (float): The amount sold 3 days before the date of sale
                4. 4_days_ago (float): The amount sold 4 days before the date of sale
                5. 5_days_ago (float): The amount sold 5 days before the date of sale
                6. 6_days_ago (float): The amount sold 6 days before the date of sale
                7. 7_days_ago (float): The amount sold 7 days before the date of sale
                8. 14_days_ago (float): The amount sold 14 days before the date of sale
                9. 28_days_ago (float): The amount sold 28 days before the date of sale
                10.182_days_ago (float): The amount sold 182 days before the date of sale
                11. 364_days_ago (float): The amount sold 364 days before the date of sale
                12. mean_last_7_days (float): The mean amount sold in the 7 days 
                        before the date of sale
                13. mean_last_14_days (float): The mean amount sold in the 14 days 
                        before the date of sale
                14. mean_last_28_days (float): The mean amount sold in the 28 days 
                        before the date of sale
                15. mean_last_364_days (float): The mean amount sold in the 364 days 
                        before the date of sale

        Returns
        -------
        X : ndarray
                2-D array with new features added
        '''

        _X = X.copy()
        signal_col = 'diff_signal_scaled'
        X_len = _X.shape[0]  # number of instances in X

        if X_len <= 7:
            # if the data size < 7, raise exception error
            raise Exception("Insufficient data size")

        # add lag features
        timesteps = 0
        for lag in self.lags:
            if X_len * self.threshold_for_rolling > lag:
                timesteps = lag
                _X[f'{lag}_days_ago'] = _X[signal_col].shift(lag, fill_value=0)

        # add roll means
        for i in self.roll_groups:
            if X_len * self.threshold_for_rolling > i:
                # means
                roll_means_arr = _X[signal_col].rolling(i).mean()
                _X[f'mean_last_{i}_days'] = roll_means_arr.shift(1)
                # max
                roll_max_arr = _X[signal_col].rolling(i).max()
                _X[f'max_last_{i}_days'] = roll_max_arr.shift(1)
                # min
                roll_min_arr = _X[signal_col].rolling(i).min()
                _X[f'min_last_{i}_days'] = roll_min_arr.shift(1)

        # drop signal column because it is a label, not a feature
        _X.drop(signal_col, axis=1, inplace=True)
        return _X, timesteps
