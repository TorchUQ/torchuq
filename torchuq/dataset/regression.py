"""
IO module for UCI datasets for regression

Much of the code is adapted from: https://github.com/aamini/evidential-regression/blob/c0823f18ff015f5eb46a23f0039f4d62b76bc8d1/data_loader.py
"""

import numpy as np
import pandas as pd
import os
import h5py
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.datasets import load_boston


dataset_names_uci = ["blog", "boston", "concrete", "crime", "energy-efficiency", "fb-comment1", "fb-comment2", "forest-fires", "mpg", "naval", "power-plant", "protein", "superconductivity", "wine", "yacht"]
dataset_names = dataset_names_uci + ["kin8nm", "medical-expenditure"]


def get_regression_datasets(name, val_fraction=0.2, test_fraction=0.2, split_seed=0, normalize=True, verbose=True):
    r"""
    Returns a UCI regression dataset in the form of numpy arrays.
    
    Arguments:
        name (str): name of the dataset
        val_fraction (float): fraction of dataset to use for validation, if 0 then val dataset will return None
        test_fraction (float): fraction of the dataset used for the test set, if 0 then test dataset will return None
        split_seed (int): seed used to generate train/test split, if split_seed=-1 then the dataset is not shuffled
        normalize (boolean): normalize the dataset to have zero mean and unit variance 
        
    Returns:
        train_dataset (torch.utils.data.Dataset): training dataset
        val_dataset (torch.utils.data.Dataset): validation dataset, None if val_fraction=0.0
        test_dataset (torch.utils.data.Dataset): test dataset, None if test_fraction=0.0 
    """
    # load full dataset

    if verbose:
        print("Loading dataset {}....".format(name))

    X, y = regression_load_funs[name]()    
    X = X.astype(np.float32)   # X should be an array of shape [num_data, feature_dim]
    y = y.astype(np.float32)

    # Randomly shuffle the dataset
    if split_seed == -1:  # Do not shuffle
        permutation = range(X.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X.shape[0])

    # Compute the size of train, val, test sets
    size_val = int(np.round(X.shape[0] * val_fraction))
    size_test = int(np.round(X.shape[0] * test_fraction))
    if size_test == 0 and test_fraction != 0:
        print("Warning: For dataset %s, the test_fraction=%f but the actual test size is zero" % (name, test_fraction))
    if size_val == 0 and val_fraction != 0:
        print("Warning: For dataset %s, the val_fraction=%f but the actual val size is zero" % (name, val_fraction)) 
    assert X.shape[0] - size_val - size_test >= 2, "Train data size has to be at least 2, maybe check that test_fraction=%f and val_fraction=%f sum to less than 1?" % (test_fraction, val_fraction)
    if verbose:
        print("Splitting into train/val/test with %d/%d/%d samples" % (X.shape[0] - size_val - size_test, size_val, size_test))
        
    # Normalize the data to have unit std and zero mean
    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale < 1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale
    
    # Extract the training set
    index_train = permutation[size_val+size_test:]
    X_train = X[index_train, :]
    y_train = y[index_train]
    if normalize:
        X_train, x_train_mu, x_train_scale = standardize(X_train)
        y_train, y_train_mu, y_train_scale = standardize(y_train)
    else:
        x_train_mu, y_train_mu = 0.0, 0.0
        x_train_scale, y_train_scale = 1.0, 1.0
    train_dataset = TensorDataset(
        torch.Tensor(X_train).type(torch.float32),
        torch.Tensor(y_train).type(torch.float32),
    )
    
    # Extract the val set is applicable
    if size_val > 0:
        index_val = permutation[:size_val]
        X_val = (X[index_val, :] - x_train_mu) / x_train_scale
        y_val = (y[index_val] - y_train_mu) / y_train_scale
        val_dataset = TensorDataset(
            torch.Tensor(X_val).type(torch.float32),
            torch.Tensor(y_val).type(torch.float32),
        )
    else:
        val_dataset = None
    
    # Extract the test set if applicable
    if size_test > 0:
        index_test = permutation[size_val:size_val+size_test]
        X_test = (X[index_test, :] - x_train_mu) / x_train_scale
        y_test = (y[index_test] - y_train_mu) / y_train_scale
        test_dataset = TensorDataset(
            torch.Tensor(X_test).type(torch.float32),
            torch.Tensor(y_test).type(torch.float32),
        )
    else:
        test_dataset = None
    if verbose:
        print("Done loading dataset {}".format(name))
    return train_dataset, val_dataset, test_dataset # , in_size, target_size, y_train_scale


#####################################
# individual data files             #
#####################################
_data_dir = os.path.join(os.path.dirname(__file__), "data")


def _load_boston():
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    X, y = load_boston(return_X_y=True)
    return X, y


def _load_powerplant():
    """
    attribute information:
    features consist of hourly average ambient variables
    - temperature (t) in the range 1.81 c and 37.11 c,
    - ambient pressure (ap) in the range 992.89-1033.30 millibar,
    - relative humidity (rh) in the range 25.56% to 100.16%
    - exhaust vacuum (v) in teh range 25.36-81.56 cm hg
    - net hourly electrical energy output (ep) 420.26-495.76 mw
    the averages are taken from various sensors located around the
    plant that record the ambient variables every second.
    the variables are given without normalization.
    """
    data_file = os.path.join(_data_dir, "uci/power-plant/Folds5x2_pp.xlsx")
    data = pd.read_excel(data_file)
    x = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return x, y


def _load_concrete():
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(_data_dir, "uci/concrete/Concrete_Data.xls")
    data = pd.read_excel(data_file)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_yacht():
    """
    Attribute Information:
    Variations concern hull geometry coefficients and the Froude number:
    1. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.
    The measured variable is the residuary resistance per unit weight of displacement:
    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    data_file = os.path.join(_data_dir, "uci/yacht/yacht_hydrodynamics.data")
    data = pd.read_csv(data_file, delim_whitespace=True, header=None)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_energy_efficiency():
    """
    Data Set Information:
    We perform energy analysis using 12 different building shapes simulated in
    Ecotect. The buildings differ with respect to the glazing area, the
    glazing area distribution, and the orientation, amongst other parameters.
    We simulate various settings as functions of the afore-mentioned
    characteristics to obtain 768 building shapes. The dataset comprises
    768 samples and 8 features, aiming to predict two real valued responses.
    It can also be used as a multi-class classification problem if the
    response is rounded to the nearest integer.
    Attribute Information:
    The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
    Specifically:
    X1    Relative Compactness
    X2    Surface Area
    X3    Wall Area
    X4    Roof Area
    X5    Overall Height
    X6    Orientation
    X7    Glazing Area
    X8    Glazing Area Distribution
    y1    Heating Load
    y2    Cooling Load
    """
    data_file = os.path.join(_data_dir, "uci/energy-efficiency/ENB2012_data.xlsx")
    data = pd.read_excel(data_file)
    X = data.values[:, :-4]
    y_heating = data.to_numpy()[:, -4]
    y_cooling = data.to_numpy()[:, -3]  # There are two dead columns in the end, remove them here
    return X, y_cooling


def _load_wine():
    """
    Attribute Information:
    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """
    data_file = os.path.join(_data_dir, 'uci/wine-quality/winequality-red.csv')
    data = pd.read_csv(data_file, sep=";")
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_kin8nm():
    """
    This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of
     this data set we have used the variant 8nm, which is known to be highly non-linear and medium noisy.
    Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo
    (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 8192 cases,
    9 attributes (0 nominal, 9 continuous).
    Input variables:
    1 - theta1
    2 - theta2
    ...
    8 - theta8
    Output variable:
    9 - target
    """
    data_file = os.path.join(_data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
    data = pd.read_csv(data_file)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_naval():
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = pd.read_csv(os.path.join(_data_dir, "uci/naval/data.txt"), delim_whitespace=True, header=None)
    X = data.to_numpy()[:, :-2]
    y_compressor = data.to_numpy()[:, -2]
    y_turbine = data.to_numpy()[:, -1]
    return X, y_turbine


def _load_protein():
    """
    Physicochemical Properties of Protein Tertiary Structure Data Set
    Abstract: This is a data set of Physicochemical Properties of Protein Tertiary Structure.
    The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

    Output variable:
        RMSD-Size of the residue.
        
    Input variables:
        F1 - Total surface area.
        F2 - Non polar exposed area.
        F3 - Fractional area of exposed non polar residue.
        F4 - Fractional area of exposed non polar part of residue.
        F5 - Molecular mass weighted exposed area.
        F6 - Average deviation from standard exposed area of residue.
        F7 - Euclidian distance.
        F8 - Secondary structure penalty.
        F9 - Spacial Distribution constraints (N,K Value).
    """
    data_file = os.path.join(_data_dir, "uci/protein/CASP.csv")
    data = pd.read_csv(data_file, sep=",")
    X = data.to_numpy()[:, 1:]
    y = data.to_numpy()[:, 0]
    return X, y


def _load_crime():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/crime/communities.data'), sep=',', header=None).iloc[:, 5:]
    data = data.replace('?', np.nan)
    data = data.dropna(thresh=len(data) - 100, axis=1)  # Drop any columns that have more than 100 nan
    data = data.dropna(axis=0)  # Drop any rows that still have nan
    X, y = data.to_numpy()[:, :-1].astype(np.float32), data.to_numpy()[:, -1].astype(np.float32)
    return X, y


def _load_superconductivity():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/superconductivity/train.csv' ))
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y 


def _load_mpg():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/mpg/auto-mpg.data'), sep='\s+', header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']) 
    data = data.replace('?', np.nan)                 # There are some missing data denoted as ?                                           
    data = data.drop('car name', axis=1)             # Uninformative feature
    data = data.dropna(axis=0)

    # Transform the origin into one-hot encoding
    origin = pd.get_dummies(data.origin, prefix='origin') 
    data = data.drop('origin', axis=1).join(origin)
    X = data.to_numpy().astype(np.float)[:, 1:]
    y = data.to_numpy().astype(np.float)[:, 0]
    return X, y


def _load_blog_feedback():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/blog/blogData_train.csv'), header=None)
    X = data.to_numpy()[:, :280]
    y = data.to_numpy()[:, 280]
    # The log scale is more appropriate because the data is very skewed, this is the experiment setup in https://arxiv.org/pdf/1905.02928.pdf
    y = np.log(y + 1)   
    return X, y


def _load_medical_expenditure():
    """ Preprocess the medical expenditure dataset, the preprocessing is based on http://www.stat.uchicago.edu/~rina/jackknife/get_meps_data.ipynb"""
    reader = np.load(os.path.join(_data_dir, 'medical-expenditure/meps_data.npz'))
    return reader['X'], reader['y']


def _load_forest_fires():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/forest-fires/forestfires.csv'))
    data.isnull().values.any()
    month = pd.get_dummies(data.month, prefix='origin') 
    data = data.drop(['month', 'day'], axis=1).join(month)

    y = data['area'].to_numpy().astype(np.float)
    y = np.log(1 + y)  # Because the dataset is skewed toward zero, transform it by log (1+x) (same as original paper)
    data = data.drop('area', axis=1)
    X = data.to_numpy()[:, :-1].astype(np.float)
    return X, y


def _load_facebook_comment1():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/facebook/Features_Variant_1.csv'), header=None)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    y = np.log(1 + y)
    return X, y


def _load_facebook_comment2():
    data = pd.read_csv(os.path.join(_data_dir, 'uci/facebook/Features_Variant_2.csv'), header=None)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    y = np.log(1 + y)
    return X, y


regression_load_funs = {
    "blog": _load_blog_feedback, 
    "boston": _load_boston,
    "concrete": _load_concrete,
    "crime": _load_crime,
    "energy-efficiency": _load_energy_efficiency,
    "fb-comment1": _load_facebook_comment1,
    "fb-comment2": _load_facebook_comment2,
    "forest-fires": _load_forest_fires, 
    "kin8nm": _load_kin8nm,
    "medical-expenditure": _load_medical_expenditure, 
    "mpg": _load_mpg, 
    "naval": _load_naval,
    "power-plant": _load_powerplant,
    "protein": _load_protein,
    "superconductivity": _load_superconductivity, 
    "wine": _load_wine,
    "yacht": _load_yacht,
}


if __name__ == "__main__":
    for k in regression_load_funs:
        X, y = regression_load_funs[k]()
        print(k, X.shape)
        print(np.isnan(X).sum(), np.isnan(y).sum())
