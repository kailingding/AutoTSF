
import pandas as pd
import numpy as np


def csv_to_df(csv, time_type='day'):
    '''
            a function that convert csv to pandas dataframe

            csv format: two columns only
                    ex. datetime | value
                            03-03-2019 | 34.2
                            04-02-2020 | 40.8
            time_type: ['day', 'hour'], default: 'day'
                    when parse dates, use different parser,
    '''
    # check time_type value
    if time_type == 'day':
        def datparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')
    elif time_type == 'hour':
        def datparse(x): return pd.datetime.strptime(x, '%Y-%m-%d %H')
    else:
        raise Exception('time_type input is invalid, only ["day", "hour"]')

    try:
        df = pd.read_csv(csv, parse_dates=['datetime'], date_parser=datparse, usecols=[
                         'datetime', 'value'])
    except:
        raise Exception(
            'Input csv only allows two columns named, ["datetime", "value"]')

    return df
