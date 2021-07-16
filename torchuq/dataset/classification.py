import numpy as np
import pandas as pd
import os
import h5py
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset


dataset_names_uci = ['adult', 'breast-cancer', 'covtype', 'digits', 'iris']
dataset_names = dataset_names_uci 
dataset_nclasses = {
    'adult': 2,
    'breast-cancer': 2,
    'covtype': 7,
    'digits': 10,
    'iris': 3, 
}


def get_classification_datasets(name, val_fraction=0.2, test_fraction=0.2, split_seed=0, normalize=True, verbose=True):
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
        n_classes (int): the number of classes in the classification task, the labels will be integers {0, 1, ..., n_classes-1} 
    """
    # load full dataset

    if verbose:
        print("Loading dataset {}....".format(name))

    X, y = classification_load_funs[name]()    
    X = X.astype(np.float32)   # X should be an array of shape [num_data, feature_dim]
    y = y.astype(np.int)

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
    else:
        x_train_mu, x_train_scale = 0.0, 1.0
    train_dataset = TensorDataset(
        torch.Tensor(X_train).type(torch.float32),
        torch.Tensor(y_train).type(torch.int),
    )
    
    # Extract the val set is applicable
    if size_val > 0:
        index_val = permutation[:size_val]
        X_val = (X[index_val, :] - x_train_mu) / x_train_scale
        y_val = y[index_val] 
        val_dataset = TensorDataset(
            torch.Tensor(X_val).type(torch.float32),
            torch.Tensor(y_val).type(torch.int),
        )
    else:
        val_dataset = None
    
    # Extract the test set if applicable
    if size_test > 0:
        index_test = permutation[size_val:size_val+size_test]
        X_test = (X[index_test, :] - x_train_mu) / x_train_scale
        y_test = y[index_test]
        test_dataset = TensorDataset(
            torch.Tensor(X_test).type(torch.float32),
            torch.Tensor(y_test).type(torch.int),
        )
    else:
        test_dataset = None
    if verbose:
        print("Done loading dataset {}".format(name))
    return train_dataset, val_dataset, test_dataset 



_data_dir = os.path.join(os.path.dirname(__file__), "data")


def _load_adult():
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(os.path.join(_data_dir, 'uci/adult/adult.data'), index_col=False, names=header)
    data = data.replace(' ?', np.nan).replace('? ', np.nan)

    data['capital-gain'] = np.log(1 + data['capital-gain'])
    data['capital-loss'] = np.log(1 + data['capital-loss'])

    y = pd.get_dummies(data['income'], prefix='income')['income_ >50K'].to_numpy().astype(np.int)
    data = data.drop('income', axis=1)

    for column in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']:
        one_hot = pd.get_dummies(data[column], prefix=column) 

        data = data.drop(column, axis=1).join(one_hot)

    X = data.to_numpy().astype(np.float32)
    return X, y


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


def _load_iris():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    return X, y


def _load_digits():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    return X, y


def _load_covtype():
    from sklearn.datasets import fetch_covtype
    X, y = fetch_covtype(return_X_y=True,  data_home=os.path.join(_data_dir, 'covtype'))
    return X, y - 1 # Covtype dataset class labels are 1-7, adjust it to 0-6


classification_load_funs = {
    'adult': _load_adult,
    'breast-cancer': _load_breast_cancer,
    'covtype': _load_covtype,
    'digits': _load_digits,
    'iris': _load_iris, 
}
