import os
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd 
import torch 

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
    

class HAM10000(Dataset):
    """ The class for the HAM10000 dataset that inherits the torch Dataset interface"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y
    

def get_ham10000(data_dir='.', test_fraction=0.2, val_fraction=0.2, split_seed=0, balance_train=True, verbose=True, input_size=224):
    """ Retrieve the HAM10000 dataset. 
    
    To use this function, download the dataset folders HAM10000_images_part_1 and HAM10000_images_part_2 and the meta-data file https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
    from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000. 
    Put them into the same folder, and point data_dir to this folder.
    
    Args:
        data_dir (str): the data folder. 
        val_fraction (float): fraction of dataset to use for validation, if 0 then val dataset will return None.
        test_fraction (float): fraction of the dataset used for the test set, if 0 then test dataset will return None.
        split_seed (int): seed used to generate train/test split. 
        balance_train (bool): if True then over-sample under-represented classes in the training set, so that all classes have approximately the same number of samples. 
        verbose (bool): if True then print additional messages. 
        input_size (int): the size of the image. 
        
    Returns:
        train_dataset (torch.utils.data.Dataset): training dataset
        val_dataset (torch.utils.data.Dataset): validation dataset, None if val_fraction=0.0
        test_dataset (torch.utils.data.Dataset): test dataset, None if test_fraction=0.0 
    """
    if not os.path.isdir(os.path.join(data_dir, 'HAM10000_images_part_1')) or not os.path.isdir(os.path.join(data_dir, 'HAM10000_images_part_2')):
            assert False, "Cannot find the HAM dataset, download the data files"
        
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}   # Get the list of all images and their full path
    if verbose:
        print("Found %d images in the data directory" % len(all_image_path))
        
    # Load the meta data
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    # df_original.head()
    
    # Filter out lesion_id's that have only one image associated with it
    df_undup = df_original.groupby('lesion_id').count()   # How many images are associated with each lesion_id
    df_undup = df_undup[df_undup['image_id'] == 1]  # Get the dataframe with only unduplicated images
    df_undup.reset_index(inplace=True)
    # print(df_undup.head())
    
    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_original.head()
    df_undup = df_original[df_original['duplicates'] == 'unduplicated'] # Filter out images that don't have duplicates
    df_dup = df_original[df_original['duplicates'] == 'duplicated'] # Filter out images that don't have duplicates
    if verbose:
        print("%d images have duplicates, and %d images do not" % (len(df_dup), len(df_undup)))
    
    # We create a val/test set using df because we are sure that none of these images have duplicates in the train set 
    assert (test_fraction + val_fraction) * len(df_original) / len(df_undup) < 1.0, "There is not enough unduplicated images for such large val/test set"
    df_val, df_test = None, None
    if val_fraction > 0:
        df_undup, df_val = train_test_split(df_undup, test_size=val_fraction * len(df_original) / len(df_undup), 
                                            random_state=split_seed, stratify=df_undup['cell_type_idx'])

    if test_fraction > 0:
        df_undup, df_test = train_test_split(df_undup, test_size=test_fraction * len(df_original) / len(df_undup), 
                                             random_state=split_seed, stratify=df_undup['cell_type_idx'])
    
    # Collect all remaining unduplicated images and all duplicated images as training set 
    df_train = pd.concat([df_dup, df_undup])
    if verbose:
        print("Size of train/val/test splits = %d/%d/%d" % (len(df_train), len(df_val) if df_val is not None else 0, len(df_test) if df_test is not None else 0))
    
    # Copy fewer class to balance the number of 7 classes
    if balance_train:
        data_aug_rate = [15,10,5,50,0,40,5]
        for i in range(7):
            if data_aug_rate[i]:
                df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)        
    if verbose:
        print("------------------- Value counts on train ------------------- ")
        #     print("Augmented the training set to balance different classes")
        print(df_train['cell_type'].value_counts())
        
        if df_val is not None:
            print("------------------- Value counts on val ------------------- ")
            print(df_val['cell_type'].value_counts())
            
                
        if df_test is not None:
            print("------------------- Value counts on test ------------------- ")
            print(df_test['cell_type'].value_counts())
            
    # These are pre-computed magic numbers to normalize the dataset
    norm_mean = [0.7630365, 0.5456421, 0.5700475]
    norm_std = [0.140928, 0.1526134, 0.16997088]


    # define the transformation of the train images.
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.RandomHorizontalFlip(), 
                                              transforms.RandomVerticalFlip(), transforms.RandomRotation(10), 
                                              transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1), 
                                              transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    # define the transformation of the val images / test images.
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(), 
                                            transforms.Normalize(norm_mean, norm_std)])

    train_dataset = HAM10000(df_train.reset_index(), transform=train_transform) # Need to reset index so we can index the entries from [0, len(df_train))
    val_dataset = HAM10000(df_val.reset_index(), transform=val_transform) if df_val is not None else None   
    test_dataset = HAM10000(df_test.reset_index(), transform=val_transform) if df_test is not None else None
    
    return train_dataset, val_dataset, test_dataset