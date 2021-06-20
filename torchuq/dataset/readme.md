## UCI dataset

We include a large suite of plug-and-play datasets. These datasets have been preprocessed to remove missing data and normalized. There is also no need to download any dataset, as they are either included in the repo, or automatically downlaoded. 

To access UCI regression datasets use the function `get_uci_regression_datasets`. For example, to access the wine dataet use 

`train_dataset, val_dataset, test_dataset = get_uci_regression_datasets('wine', val_fraction=0.2, test_fraction=0.2)`

The returned dataset is a subclass of `torch.utils.data.Dataset` so you can immediately use them in the pytorch training pipeline. If you do not want to split out the test dataset (or val dataset) then set test_fraction (or val_fraction) to 0. The function will return ``None`` instead of a `Dataset` class. 

The available datasets include: 
[wine](https://archive.ics.uci.edu/ml/datasets/wine+quality)
[boston](https://www.kaggle.com/heptapod/uci-ml-datasets) 
[concrete](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
[power-plant](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) 
[yacht](https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics)
[energy-efficiency](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)  
[naval](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)
[protein](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure)
[crime](http://archive.ics.uci.edu/ml/datasets/communities+and+crime)

We also include datasets that are not in the UCI repository, but for convenience are also included under the same interface
[kin8nm](https://www.openml.org/d/189) 
