# ==========================
# Author: Kailing Ding, Shen Hu
# Created Date: 05/05/2020
# ==========================

# ==========================
# Author: Kailing Ding, Shen Hu
# Created Date: 05/05/2020
# ==========================
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
    

    
'''Naive test using the statsmodels.api.tsa.seasonal_decompose
X is an n*2 dataframe with one datetime column and one data column'''    
def naiveTest(X):
    data = X.set_index(["datetime"])
    data = data.astype("int32")
    result = seasonal_decompose(data, model='additive', freq=1)
    result.plot()

    
    
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)
'''Using stl decomposition to check trends, seasonality
X is an n*2 dataframe with one datetime column and one data column
https://github.com/jrmontag/STLDecompose check this website for downloading the stldecompose package
'''
def stldecom(X,period_1):
    data = X.set_index(["datetime"])
    decomp = decompose(data, period = period_1)
    decomp.plot()


