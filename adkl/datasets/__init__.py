import json
from glob import glob
from hashlib import sha256
from itertools import product
from os.path import join, dirname, exists, realpath
from random import Random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from torchvision.transforms.functional import to_tensor

from .metadataset import MetaDataLoader
from .mutlitask_dataset import MultiTaskDataLoader

DATASETS_ROOT = join(dirname(dirname(dirname(realpath(__file__)))), 'datasets')


def transform(x, y, x_to_float=True, filename=''):
    """
    Transforms x and y to float, removing NaNs in the process.

    Parameters
    ----------
    x, y: np.array
        The inputs and targets
    x_to_float: bool

    Returns
    -------
    x, y: np.array
    """
    x, y = np.array(x), np.array(y)

    try:
        x, y = x, y.astype('float32')
    except ValueError:
        print(f'ValueError with {filename}')
        y[y == 'True'] = '1'
        y[y == 'False'] = '0'
        x, y = x, y.astype('float32')

    # Removes NaN values
    finite = np.isfinite(y)

    x, y = x[finite], y[finite]

    if x_to_float:
        x = x.astype('float32')
        x = x.reshape(-1, 1)

    y = y.reshape(-1, 1)

    return x, y


def episode_loader_molecules(filename, skip=0, y_scaler=None):
    # print(y_scaler)
    with open(filename, 'r') as f_in:
        x, y = zip(*([line[:-1].split('\t') for line in f_in][skip:]))
        y = list(map(float, y))
    x, y = np.array(x), np.array(y).reshape((-1, 1)).astype('float32')
    if y_scaler == 'minmax':
        y = MinMaxScaler().fit_transform(y).astype('float32')
    elif y_scaler == 'robust':
        y = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(y).astype('float32')
    elif y_scaler == 'normal':
        y = QuantileTransformer(random_state=0).fit_transform(y).astype('float32')
    elif y_scaler == 'log':
        y = np.log(y - y.min() + 1).astype('float32')
    elif y_scaler == 'minmax.log':
        y = np.log(y - y.min() + 1).astype('float32')
        y = MinMaxScaler().fit_transform(y).astype('float32')
    else:
        pass
    return x, y, None, None


def episode_loader_harmonics(filename, skip=3, y_scaler=None):
    with open(filename, 'r') as f_in:
        for _ in range(skip):
            f_in.readline()
        x, y = zip(*[line[:-1].split(',') for line in f_in])

    x, y = transform(x, y)

    return x, y, None, None


def __get_train_test_files(dataset_name, ds_folder=None, max_tasks=None, test_size=0.25):
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, dataset_name)

    DATASETS_MAPPING = dict(
        antibacterial=dict(files=[join(ds_folder, x) for x in glob("{}/*.txt".format(ds_folder))]),
        bindingdb=dict(files=[join(ds_folder, x) for x in glob("{}/*.tsv".format(ds_folder))]),
        sinusoidals=dict(files=[join(ds_folder, x) for x in glob("{}/*.csv".format(ds_folder))]),
    )

    if dataset_name not in DATASETS_MAPPING:
        raise Exception(f"Unhandled dataset. The name of \
            the dataset should be one of those: {list(DATASETS_MAPPING.keys())}")

    splitting_file = join(ds_folder, dataset_name + '.json')
    if exists(splitting_file):
        with open(splitting_file) as fd:
            temp = json.load(fd)
        if max_tasks is not None:
            max_tasks_train = int(0.75 * max_tasks)
            max_tasks_test = int(max_tasks - max_tasks_train)
        else:
            max_tasks_train, max_tasks_test = None, None
        train_files = [join(dirname(ds_folder), f) for f in temp['Dtrain']][:max_tasks_train]
        test_files = [join(dirname(ds_folder), f) for f in temp['Dtest']][:max_tasks_test]
    else:
        all_files = DATASETS_MAPPING[dataset_name]['files'][:max_tasks]
        train_files, test_files = train_test_split(all_files, test_size=test_size)

    print('number of train tasks:', len(train_files))
    print('number of test tasks:', len(test_files))
    return train_files, test_files


def __get_episode_loader(dataset_name):
    DATASETS_MAPPING = dict(
        antibacterial=(lambda x, y: episode_loader_molecules(x, skip=0, y_scaler=y)),
        binding=(lambda x, y: episode_loader_molecules(x, skip=2, y_scaler=y)),
        sinusoidals=(lambda x, y: episode_loader_harmonics(x, skip=3, y_scaler=y)),,
    )

    if dataset_name not in DATASETS_MAPPING:
        raise Exception(f"Unhandled dataset. The name of \
            the dataset should be one of those: {list(DATASETS_MAPPING.keys())}")
    return DATASETS_MAPPING.get(dataset_name.lower())


def get_dataloader(dataset_name, ds_folder=None, max_tasks=None, test_size=0.25, **dataset_params):
    dataset_type, dataset_name = dataset_name.split('-')
    train_files, test_files = __get_train_test_files(dataset_name, ds_folder, max_tasks, test_size=test_size)
    is_molecule_dataset = (dataset_name in ['bindingdb', 'antibacterial'])
    dt = MetaDataLoader(train_files=train_files, test_files=test_files,
                        episode_loader=__get_episode_loader(dataset_name),
                        is_molecule_dataset=is_molecule_dataset,
                        **dataset_params)
    return dt
