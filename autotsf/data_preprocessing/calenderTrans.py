
# ==========================
# Author: Kailing Ding
# Created Date: 05/05/2020
# ==========================

import numpy as np
import pandas as pd
import datetime


class CalenderTransformer:
    '''
        A column transformer that adds calender-related features to X
    '''

    def __init__(self):
        return None

    def _get_day_info(self, date):
        '''
        Get day of the week and return an integer

        Weekday to Integer conversion

        Mon | Tues | Wed | Thurs | Fri | Sat | Sun
         0  |  1   |  2  |   3   |  4  |  5  |  6

        Parameters
        ---------- 
                date : datetime object
                    datetime input 

        Returns
        -------
                _weekday : int
                        day of the week as an integer

                _day_of_year : int
                        day of the year as an integer

                _week_of_year : int
                        week of the year as an integer
        '''
        if not isinstance(date, datetime.datetime):
            raise TypeError("Date is not a datetime.datetime object")

        # generate date information
        _weekday = date.weekday()
        _day_of_year = date.timetuple().tm_yday / 365
        _week_of_year = date.isocalendar()[1] / 53

        return pd.Series({'weekday': _weekday, 'day_of_year': _day_of_year,
                          'week_of_year': _week_of_year})

    def transform(self, X, y=None, ):
        '''
                Adding calender related features to the training set

        Parameters
        ----------
            X : ndarray
                    Input array
            y : n x 1 array
                    label arr

        Feature Engineering
        -------------------
        List of variable added to the training set:
                1. day_of_week (int): The day of the week of the date of sale
                2. day_of_year (int): The day of the year of the date of sale
                3. week_of_year (int): The week of the year of the date of sale
                4. is_monday (bool): If the date of sale was a Monday
                5. is_tuesday (bool): If the date of sale was a Tuesday
                6. is_wednesday (bool): If the date of sale was a Weddnesday
                7. is_thursday (bool): If the date of sale was Thursday
                8. is_friday (bool): If the date of sale was Friday
                9. is_saturday (bool): If the date of sale was Saturday
                10. is_sunady (bool): If the date of sale was Sunday
                11. holiday_today (bool): If the date of sale was a holiday
                12. holiday_tmr (bool): If the date after the date of sale was a holiday
                13. holiday_yesterday (bool): If the date before the date of sale was a holiday

        Returns
        -------
        X : ndarray
                output array
        '''
        _X = X.copy()
        _X = pd.concat([_X, _X['datetime'].apply(self._get_day_info)], axis=1)

        # get weekday boolean columns
        weekday_df = pd.get_dummies(_X['weekday'], prefix='is_weekday_')
        _X = pd.concat([_X, weekday_df], axis=1)

        # TODO: add holiday information to the datasets
        _X.drop(['datetime', 'weekday'], axis=1, inplace=True)
        return _X


if __name__ == "__main__":
    calenderTrans = CalenderTransformer()
    ca_0 = pd.read_csv("../sample_data/data/CA_0.csv", parse_dates=['datetime'])
    print(calenderTrans.transform(ca_0).columns)
