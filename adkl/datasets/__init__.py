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

MINIMUM_DENSITY = 1e-2

DESCRIPTIONS = dict(
    pca=pd.read_csv(join(DATASETS_ROOT, 'synthetic', 'description_pca.csv'), index_col=0),
    descriptors=pd.read_csv(join(DATASETS_ROOT, 'synthetic', 'description_descriptors.csv'), index_col=0),
)

with open(join(DATASETS_ROOT, 'synthetic', 'index.csv'), 'r') as f:
    INDEX = np.array([line[:-1] for line in f])


def make_sparse(filename, x, y, density):
    """
    Selects only a fraction of the examples that will be fed to the model.
    To make sure the sample is the same every time, we use a pseudo random number generator
    that is seeded using a hash of the filename.

    Parameters
    ----------
    filename: str
        The path of the file (task).
    x: np.array
        The inputs
    y: np.array
        The targets
    density: float
        The density, between MINIMUM_DENSITY and 1

    Returns
    -------
    x, y: np.array
        The inputs and targets.
    """
    if density == 1:
        return x, y

    seed = int(sha256(filename.encode()).hexdigest()[:16], 16)
    rng = Random(seed)

    n = len(x)
    k = int(density * n)

    indices = np.array(rng.sample(list(range(n)), k)).astype(int)

    x = x[indices]
    y = y[indices]

    return x, y


def select_subset(x, y, density):
    """
    We want the same overall number of training examples,
    regardless of the density. That means that for higher densities, we need to sample from
    fewer examples.

    Parameters
    ----------
    x, y: np.array
        The inputs and targets
    density: float
        The density.

    Returns
    -------
    x, y: np.array
        The transformed inputs and targets.
    """
    assert density < 0 or MINIMUM_DENSITY <= density <= 1

    if density == MINIMUM_DENSITY:
        return x, y

    rng = Random(0)
    k = int(len(x) * MINIMUM_DENSITY / density)
    indices = rng.sample(list(range(len(x))), k)

    return np.array(x)[indices], np.array(y)[indices]


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


def episode_loader_molecules(filename, skip=0, y_scaler=None, density=-1):
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


def episode_loader_harmonics(filename, skip=3, y_scaler=None, density=-1):
    with open(filename, 'r') as f_in:
        for _ in range(skip):
            f_in.readline()
        x, y = zip(*[line[:-1].split(',') for line in f_in])

    x, y = transform(x, y)

    if density > 0:
        # We select a subset of the examples, so that if the density is not 100%
        # there are still as many examples.
        select_subset(x, y, density)
        x, y = make_sparse(filename, x, y, density)

    return x, y, None, None


def episode_loader_synthetic(filename, skip=3, y_scaler=None, density=-1):

    y = np.memmap(filename, dtype='float32', mode='r')
    x = INDEX

    scaler = None

    if y_scaler is not None:

        task = filename.split('/')[-1].split('__')[2][:-4]

        if 'synthetic/descriptors' in filename:
            description = DESCRIPTIONS['descriptors']
        elif 'synthetic/pca' in filename:
            description = DESCRIPTIONS['pca']

        if y_scaler == 'minmax':
            minimum, maximum = description.loc[int(task), ['min', 'max']]
            scale = maximum - minimum

            scaler = lambda y: (y - minimum) / scale
        elif y_scaler == 'log':
            minimum = description.loc[int(task), 'min']
            scaler = lambda y: np.log(y - minimum + 1)

    return x, y, None, scaler


def episode_loader_image(filename, density=-1, **kwargs):
    """
    Loads an image and changes it to a task.

    Parameters
    ----------
    filename
    kwargs
    density

    Returns
    -------
    x: np.array
        The array of all (28x28, in the case of MNIST) possible coordinates
    y: np.array

    """

    label = filename.split('/')[-2]
    label = None if label == 'none' else np.array([float(label)])

    with Image.open(filename) as image:
        array = to_tensor(image).numpy()

    h, w = array.shape[-2:]

    x = np.array([[x1, x2] for x1, x2 in product(range(h), range(w))])
    y = np.array([array[..., x1, x2] for x1, x2 in x])

    if density > 0:
        # We select a subset of the examples, so that if the density is not 100%
        # there are still as many examples.
        select_subset(x, y, density)
        x, y = make_sparse(filename, x, y, density)

    return x, y, label, None


def __get_train_test_files(dataset_name, ds_folder=None, max_tasks=None, test_size=0.25):
    if ds_folder is None:
        if dataset_name in {'descriptors', 'pca'}:
            ds_folder = join(DATASETS_ROOT, 'synthetic', dataset_name)
        else:
            ds_folder = join(DATASETS_ROOT, dataset_name)

    DATASETS_MAPPING = dict(
        antibacterial=dict(files=[join(ds_folder, x) for x in glob("{}/*.txt".format(ds_folder))]),
        bindingdb=dict(files=[join(ds_folder, x) for x in glob("{}/*.tsv".format(ds_folder))]),
        toy=dict(files=[join(ds_folder, x) for x in glob("{}/*.csv".format(ds_folder))]),
        easytoy=dict(files=[join(ds_folder, x) for x in glob("{}/*.csv".format(ds_folder))]),
        hardtoy=dict(files=[join(ds_folder, x) for x in glob("{}/*.csv".format(ds_folder))]),
        descriptors=dict(files=[join(ds_folder, x) for x in glob("{}/*.csv".format(ds_folder))]),
        pca=dict(files=[join(ds_folder, x) for x in glob("{}/*.csv".format(ds_folder))]),
        synthetic=dict(files=[join(ds_folder, x) for x in glob("{}/*/*.dat".format(ds_folder))]),
        mnist=dict(files=[join(ds_folder, x) for x in glob("{}/*/*/*.png".format(ds_folder))]),
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
    elif dataset_name in {'mnist'}:
        if max_tasks is not None:
            max_tasks_train = int(0.75 * max_tasks)
            max_tasks_test = int(max_tasks - max_tasks_train)
        else:
            max_tasks_train, max_tasks_test = None, None
        train_files = [join(ds_folder, x) for x in glob("{}/train/*/*.png".format(ds_folder))][:max_tasks_train]
        test_files = [join(ds_folder, x) for x in glob("{}/test/*/*.png".format(ds_folder))][:max_tasks_test]
    else:
        all_files = DATASETS_MAPPING[dataset_name]['files'][:max_tasks]
        train_files, test_files = train_test_split(all_files, test_size=test_size)

    print('number of train tasks:', len(train_files))
    print('number of test tasks:', len(test_files))
    return train_files, test_files


def __get_episode_loader(dataset_name, density=-1):
    DATASETS_MAPPING = dict(
        antibacterial=(lambda x, y: episode_loader_molecules(x, skip=0, y_scaler=y, density=density)),
        bindingdb=(lambda x, y: episode_loader_molecules(x, skip=2, y_scaler=y, density=density)),
        toy=(lambda x, y: episode_loader_harmonics(x, skip=3, y_scaler=y, density=density)),
        easytoy=(lambda x, y: episode_loader_harmonics(x, skip=3, y_scaler=y, density=density)),
        hardtoy=(lambda x, y: episode_loader_harmonics(x, skip=3, y_scaler=y, density=density)),
        mnist=(lambda x, y: episode_loader_image(x, density=density)),
        descriptors=(lambda x, y: episode_loader_synthetic(x, skip=1, y_scaler=y,density=density)),
        pca=(lambda x, y: episode_loader_synthetic(x, skip=1, y_scaler=y, density=density)),
        synthetic=(lambda x, y: episode_loader_synthetic(x, skip=1, y_scaler=y, density=density)),
    )

    if dataset_name not in DATASETS_MAPPING:
        raise Exception(f"Unhandled dataset. The name of \
            the dataset should be one of those: {list(DATASETS_MAPPING.keys())}")
    return DATASETS_MAPPING.get(dataset_name.lower())


def get_dataloader(dataset_name, ds_folder=None, max_tasks=None, test_size=0.25, density=-1, **dataset_params):
    dataset_type, dataset_name = dataset_name.split('-')
    train_files, test_files = __get_train_test_files(dataset_name, ds_folder, max_tasks, test_size=test_size)
    is_molecule_dataset = (dataset_name in ['bindingdb', 'antibacterial', 'descriptors', 'pca',
                                            'coherent_descriptors', 'synthetic'])
    if dataset_type.lower() == "meta":
        dt = MetaDataLoader(train_files=train_files, test_files=test_files,
                            episode_loader=__get_episode_loader(dataset_name, density=density),
                            is_molecule_dataset=is_molecule_dataset,
                            **dataset_params)
    elif dataset_type.lower() == "multi":
        dt = MultiTaskDataLoader(train_files=train_files, test_files=test_files,
                                 episode_loader=__get_episode_loader(dataset_name, density=density),
                                 is_molecule_dataset=is_molecule_dataset,
                                 **dataset_params)
    else:
        raise NotImplementedError
    return dt
