import pandas as pd
from autotsf.data_preprocessing.salesHistoryTrans import SalesHistoryTransformer
from autotsf.utils._testing import get_sample_data

def test_salesTrans():
    salesTransformer = SalesHistoryTransformer()
    sample_data = pd.DataFrame({"diff_signal_scaled": [1, 3, 5, 6, 8, 10, 20, 100]})
    X, max_lag = salesTransformer.transform(sample_data)
    assert X.iloc[-1, 0] == 20
    assert X.iloc[-1, 1] == 10
    assert max_lag == 2


if __name__ == "__main__":
    test_salesTrans()