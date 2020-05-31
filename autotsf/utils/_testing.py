import pandas as pd

def get_sample_data(num_samples=1):
    samples = []
    for i in range(1, num_samples+1):
        try:
            sample_data = pd.read_csv(f"autotsf/sample_data/sample_data_{i}.csv",
                                      parse_dates=['datetime'])
            samples.append(sample_data)
        except:
            break
    return samples