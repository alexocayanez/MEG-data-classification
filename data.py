import os
import pathlib
from typing import Tuple

from filelock import FileLock
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

LABEL_NAMES = ["rest", "task_motor", "task_story_math", "task_working_memory"]

def load_file(file_name_path: str) -> Tuple[np.ndarray, int]:
    with h5py.File(file_name_path, "r") as f:
        key = list(f.keys())[0]
        data = f.get(key)[()]

    # Extract label as a string from filename
    # Option 1
    """ label = "_".join(file_name_path.split("/")[-1].split("_")[:-2]) """
    # Option 2 (more flexible)
    for label_name in LABEL_NAMES:
        if label_name in str(file_name_path):
            label = label_name

    return data, LABEL_NAMES.index(label) 

def normalize(data: np.ndarray) -> np.ndarray:
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def standardize(data: np.ndarray) -> np.ndarray:
    return (data - data.mean()) / data.std()

def preprocess_meg_sample(  data: np.ndarray, 
                            normalization: str="z_score", 
                            time_wise_normalization: bool = True
                         ) -> np.ndarray:
    assert normalization in ["z_score", "min_max"]

    if time_wise_normalization:
        data = data.reshape(data.shape[1], data.shape[0])
        for i in range(data.shape[0]):
            if normalization == "min_max":
                data[i] = normalize(data[i])
            elif normalization == "z_score":
                data[i] = standardize(data[i])
        processed_data = data.reshape(data.shape[1], data.shape[0])

    else:
        if normalization == "min_max":
            processed_data = normalize(data)
        elif normalization == "z_score":
            processed_data = standardize(data)
    # data downsampling
    processed_data = processed_data[:, ::2]
    return processed_data

class MEGDataset(Dataset):
    def __init__(self, path_to_dir: str, normalization: str="z_score", time_wise_normalization: bool = True) -> None:
        self.path_to_dir = path_to_dir
        self.absolute_path = os.path.abspath(path_to_dir)
        self.directory = pathlib.Path(self.absolute_path)
        self.normalization = normalization
        self.time_wise_normalization = time_wise_normalization

    def __len__(self):  # returns number of files in directory
        """ return len(os.listdir(self.path_to_dir)) """
        return len(list(self.directory.glob('*.h5')))
    
    def __getitem__(self, index):
        #file_name_path = os.path.join(self.path_to_dir, os.listdir(self.path_to_dir)[index])
        file_name_path = list(self.directory.glob('**/*.h5'))[index]

        data, label = load_file(file_name_path)
        data = preprocess_meg_sample(data, normalization=self.normalization, time_wise_normalization=self.time_wise_normalization)

        sample = {
            "data" : torch.transpose(torch.tensor(data, dtype=torch.float32), 0, 1),
            "label" : torch.tensor(label)
        }

        return sample["data"], sample["label"]
    
def get_datasets(type: str="intra-subject"):
    if type == "intra-subject":
        with FileLock(os.path.join("data/Final Project data/Intra/train", ".ray.lock")):
            dataset_train = MEGDataset("data/Final Project data/Intra/train")
        with FileLock(os.path.join("data/Final Project data/Intra/train", ".ray.lock")):
            dataset_test = MEGDataset("data/Final Project data/Intra/test")
        return dataset_train, dataset_test
    elif type == "cross-subject":
        with FileLock(os.path.join("data/Final Project data/Cross/train", ".ray.lock")):
            dataset_train = MEGDataset("data/Final Project data/Cross/train")
        with FileLock(os.path.join("data/Final Project data/Cross/test1", ".ray.lock")):
            dataset_test_1 = MEGDataset("data/Final Project data/Cross/test1")
        with FileLock(os.path.join("data/Final Project data/Cross/test2", ".ray.lock")):
            dataset_test_2 = MEGDataset("data/Final Project data/Cross/test2")
        with FileLock(os.path.join("data/Final Project data/Cross/test3", ".ray.lock")):
            dataset_test_3 = MEGDataset("data/Final Project data/Cross/test3")
        dataset_test = ConcatDataset([dataset_test_1, dataset_test_2, dataset_test_3])
        return dataset_train, dataset_test
    else:
        raise NameError("Type of data incorrectly specified.")

def get_dataloaders(type: str="intra-subject", batch_size: int=2) -> Tuple[DataLoader, DataLoader]:
    if type in ["intra-subject", "cross-subject"]:
        train, test = get_datasets(type=type)
        return DataLoader(train, batch_size=batch_size), DataLoader(test, batch_size=batch_size)
    else:
        raise NameError("Type of data incorrectly specified.")


