from autotsf.flows.autotsf import AutoTSF
from autotsf.utils._testing import get_sample_data


def test_autotsf():
    auto_ts = AutoTSF()
    sample_data = get_sample_data(num_samples=4)[2]
    X = sample_data.iloc[:-4, :]
    X_test = sample_data.iloc[-4:, :]
    auto_ts.train(X)
    pred_df = auto_ts.forecast(num_step_forecast=4)
    print(pred_df)
    print(X_test)


if __name__ == "__main__":
    test_autotsf()