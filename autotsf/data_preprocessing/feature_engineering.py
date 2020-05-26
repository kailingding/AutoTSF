# ==========================
# Author: Kailing Ding, Shen Hu
# Updated Date: 05/09/2020
# ==========================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def get_holiday(X, holiday_df):
    '''
		Append a column to X that indicates the holiday info on a specific date
		
	Parameters
	----------
	X : array-like
		Input data
	holiday_df: n x 2 array-like
		The data contains information about its date

	Returns
	-------
	X_ : array_like
		a copy of X that combine X and holiday_df(without data)
    '''
    merged = X.merge(holiday_df,how = "inner", on = 'datetime')
    lab = LabelEncoder()
    lab.fit(merged["holiday"])
    labels = lab.transform(merged["holiday"])
    merged["holiday"] = labels
    enc = OneHotEncoder()
    enc.fit(merged["holiday"].values.reshape(-1, 1))
    matrix = enc.transform(merged["holiday"].values.reshape(-1, 1))
    dense = matrix.todense()
    for i in range(len(enc.categories_[0])):
        merged["holiday" + str(enc.categories_[0][i])] = dense[:,i]
    merged.drop(axis = 1, labels = ["data","holiday","event"], inplace = True)
    return merged

def get_events(X,event_df):
    '''
    	Append a column to X that indicates the event info on a specific date

    Parameters
	----------
	X : array-like
		Input data
	event_df: n x 2 array-like
		The data contains information about its date

	Returns
	-------
	X_ : array_like
		a copy of X that combine X and event_df(without data)
		
    '''
    merged = X.merge(event_df,how = "inner", on = "datetime")
    lab = LabelEncoder()
    lab.fit(merged["event"])
    labels = lab.transform(merged["event"])
    merged["event"] = labels
    enc = OneHotEncoder()
    enc.fit(merged["event"].values.reshape(-1, 1))
    matrix = enc.transform(merged["event"].values.reshape(-1, 1))
    dense = matrix.todense()
    for i in range(len(enc.categories_[0])):
        merged["event" + str(enc.categories_[0][i])] = dense[:,i]
    merged.drop(axis = 1, labels = ["data","holiday","event"], inplace = True)
    return merged
