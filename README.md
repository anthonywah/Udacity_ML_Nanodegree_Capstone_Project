# Udacity ML Engineer Nanodegree Capstone Project
### Predicting Bitcoin Movement Using MLP
_\*Developed by Wah Chi Shing, Anthony\*_

This project aims at predicting short-term movement in BTCUSDT prices. A neural network algorithm, MLP is used and multiple alternatives versions are also included. 

---

The structure of this directory looks like below:

```
|-input_data
|  |-BTCUSDT_2020-09-11.csv
|  |-BTCUSDT_2020-11-06.csv
|  |- ...
|
|-processed_data
|  |-test.csv
|  |-train.csv
|
|-source
|  |-predict.py
|  |-train.py
|  |-model.py
|
|-binance_data_calling.py
|-plotting_helper.py
|-data_processing.py
|-data_reader.py
|-model_helper.py
|-features_helper.py
|
|-1_data_preparation_and_exploration.ipynb
|-2_benchmark_model_computation.ipynb
|-3_target_features_processing.ipynb
|-4_target_model_training_and_evaluation.ipynb
|-5_model_improvement_alternatives.ipynb
|-6_Summary.ipynb
|-Ext_features_formula.ipynb
|-1_data_preparation_and_exploration.html
|-2_benchmark_model_computation.html
|-3_target_features_processing.html
|-4_target_model_training_and_evaluation.html
|-5_model_improvement_alternatives.html
|-6_Summary.html
|-Ext_features_formula.html
|
|-README.md
|-Udacity_ML_Nanodegree_Capstone_Project.pdf
|
|-summary_dict.json
```

`input_data` contains all open-high-low-close-volume data of BTCUSDT prices from Binance in 2020, called by `binance_data_calling.py`.

`processed_data` contains organised data for training/prediction.

`source` provides scripts for neural network training/model instance building. 

`data_processing.py`, `data_reader.py` contains helper functions that help with data I/O and pre-processing.

`features_helper.py` contains wrapper class for all features-generating formula.

`model_helper.py` contains wrapper class for different Sagemaker estimators construction.

`plotting_helper.py` contains a simple function for visualizing prediction result.

`summary_dict.json` contains a summary of result to be explained in `6_Summary.ipynb`

As for all the other notebooks, `*.ipynb` contains the model development progress, as noted by the step number and description included in the file name.

It is recommended that one starts reading this project from `Udacity_ML_Nanodegree_Capstone_Project.pdf` to know more about Bitcoin price movements, and then `1_data_preparation_and_exploration.ipynb` and subsequently other notebooks to understand more thoroughly how this project is built. 

---

### REMARKS:

- In `5_model_improvement_alternatives.ipynb` I tried codes provided in Module 3, Boston Housing example for XGBoost algorithm, but it doesn't work as it always throws me an "can't set attribute error" at the `self.predictor.content_type = 'text/csv'` line. I decided to comment that block for the time being as I realised that it is taking too much time for me on this project, and further debugging will cost me another month of subscription fee.



