# AutoTSF

Automated time-series forecast pipeline with state-of-art machine learning and deep learning algorithm. 🚀🚀🚀

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AutoTSF.

```bash
$ pip install autotsf
```

## Getting Started 

```python
from autotsf import AutoTSF               # Import

auto_tsf = AutoTSF()                      # Initialization
auto_tsf.train(data)                      # automated feature engineering and model training
pred_df = auto_tsf.predict(num_steps=7)   # one week forecast
```

## Authors

* **Kailing Ding** - [Github](https://github.com/kailingding)
* **Yuxuan Fan** - [Github](https://github.com/991231/)
* **Jerry Chan**
* **Shen Hu**
* **Jeffrey Chen**

See also the list of [contributors](https://github.com/kailingding/Autotsf/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
