# ==========================
# Author: Kailing Ding, Shen Hu
# Created Date: 05/05/2020
# ==========================

# ==========================
# Author: Kailing Ding, Shen Hu
# Created Date: 05/05/2020
# ==========================

'''
Seasonality test using periodogram: the seasonality 
'''

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


'''
Use periodogram to test seasonality. The seasonality is most pronounced at the peak frequencies.
X is an n*2 dataframe with one datetime column and one data column
'''
def testSeasonality(X):
    fs = len(X)
    x = X["data"]
    f, Pxx_den = signal.periodogram(x,fs)
    plt.plot(f, abs(Pxx_den))
    

