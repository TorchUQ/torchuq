## Regression datasets

We include a large suite of plug-and-play datasets. These datasets have been preprocessed to remove missing data and normalized. There is also no need to download any dataset, as they are either included in the repo, or automatically downlaoded. 

To access these regression datasets use the function `regression.get_regression_datasets`. For example, to access the wine dataet use 

`train_dataset, val_dataset, test_dataset = get_regression_datasets('wine', val_fraction=0.2, test_fraction=0.2)`

The returned dataset is a subclass of `torch.utils.data.Dataset` so you can immediately use them in the pytorch training pipeline. If you do not want to split out the test dataset (or val dataset) then set test_fraction (or val_fraction) to 0. The function will return ``None`` instead of a `Dataset` class. 

The available datasets include: 
[blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback)
[boston](https://www.kaggle.com/heptapod/uci-ml-datasets) 
[concrete](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
[crime](http://archive.ics.uci.edu/ml/datasets/communities+and+crime)
[energy-efficiency](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)
[fbcomment-1](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset)
[fbcomment-2](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset)
[forest-fires](http://archive.ics.uci.edu/ml/datasets/Forest+Fires)
[mpg](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
[naval](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)
[power-plant](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) 
[protein](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure)
[superconductivity](https://archive.ics.uci.edu/ml/datasets/superconductivty+data)
[wine](https://archive.ics.uci.edu/ml/datasets/wine+quality)
[yacht](https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics)

Some datasets are not in the UCI repository, but for convenience we include them under the same interface
[kin8nm](https://www.openml.org/d/189) 
[medical-expenditure](https://meps.ahrq.gov/data_stats/download_data_files.jsp)


## Classification datasets

[adult](https://archive.ics.uci.edu/ml/datasets/Adult)
[breast-cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
[covtype](https://archive.ics.uci.edu/ml/datasets/covertype)
[digits](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)
[iris](https://archive.ics.uci.edu/ml/datasets/iris)

## Skin Legion classification 

To access the [HAM10000](https://www.nature.com/articles/sdata2018161) dataset