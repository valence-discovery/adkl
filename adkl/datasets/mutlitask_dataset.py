import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from adkl.feature_extraction import (FingerprintsTransformer, AdjGraphTransformer, SequenceTransformer)
from adkl.feature_extraction.constants import SMILES_ALPHABET, ATOM_LIST
from .metadataset import MoleculeMetaDataset, MetaRegressionDataset


TRANSFORMER_MAPPING = {
    'FingerprintsTransformer': FingerprintsTransformer(kind="ecfp2"),
    'AdjGraphTransformer': AdjGraphTransformer(),
}


def custom_collate(x):
    res = list(zip(*x))
    return res[0], res[1]


class MultiTaskDataset(Dataset):
    INF = int(1e9)

    def __init__(self, tasks_filenames, episode_loader, use_graph=False,
                 max_examples_per_episode=None, seed=42, is_train=True,
                 valid_size=0.25, is_molecular_dataset=False):
        self.episode_loader = episode_loader
        self.tasks_filenames = tasks_filenames[:]
        self.max_examples_per_episode = max_examples_per_episode
        self.rgn = np.random.RandomState(seed)
        self.use_graph = use_graph
        if is_molecular_dataset:
            if self.use_graph:
                transformer = AdjGraphTransformer()
            else:
                transformer = SequenceTransformer(SMILES_ALPHABET, returnTensor=True)
            self.x_transformer = lambda x: transformer.transform(x)
        else:
            self.x_transformer = lambda x: torch.FloatTensor(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.vocab = SMILES_ALPHABET if not use_graph else ATOM_LIST
        self.init_tasks_sizes()
        self.is_train = is_train
        self.valid_size = valid_size
        self._n_tasks = len(self.tasks_filenames)

    @property
    def n_tasks(self):
        return self._n_tasks

    def __len__(self):
        return self.INF

    @staticmethod
    def cuda_tensor(x, use_available_gpu=True):
        if isinstance(x, torch.Tensor) and torch.cuda.is_available() and use_available_gpu:
            x = x.cuda()
        return x

    def init_tasks_sizes(self):
        def file_len(fname):
            with open(fname) as f:
                for i, _ in enumerate(f):
                    pass
            return i + 1
        self.task_sizes = np.array([file_len(f) for f in self.tasks_filenames])

    def __getitem__(self, i):
        file_index = i % len(self.tasks_filenames)
        x, y, _ = self.episode_loader(self.tasks_filenames[file_index])
        if self.is_train:
            x, _, y, _ = train_test_split(x, y, test_size=self.valid_size)
        else:
            _, x, _, y = train_test_split(x, y, test_size=self.valid_size)
        n, indexes = len(x), np.arange(len(x))

        self.rgn.shuffle(indexes)
        k = min(n, self.max_examples_per_episode)
        indexes = indexes[:k]
        y_train = y[indexes]
        x_train = x[indexes]
        return self._episode(x_train, y_train, file_index)

    def _episode(self, xtrain, ytrain, idx=None):
        x = self.cuda_tensor(self.x_transformer(xtrain))
        y = self.cuda_tensor(self.y_transformer(ytrain))
        idx = idx
        return (x, idx), y

    def get_sampling_weights(self):
        x = np.log2(self.task_sizes)
        return x / x.sum()


class MultiTaskDataLoader():
    def __init__(self, train_files, test_files, batch_size=32, valid_size=0.25,
                 steps_per_epoch=1000, is_molecular_dataset=False, **kwargs):
        # print(kwargs)
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.kwargs = kwargs
        self.nb_workers = 0  # os.cpu_count()
        self.train_files = train_files
        self.test_files = test_files
        self.spe = steps_per_epoch
        self.imd = is_molecular_dataset
        self.dataset_cls = MoleculeMetaDataset if is_molecular_dataset else MetaRegressionDataset

    def load_partitions(self):
        def collate(x):
            return list(zip(*x))

        train = MultiTaskDataset(self.train_files, is_train=True, valid_size=self.valid_size, is_molecular_dataset=self.imd, **self.kwargs)
        valid = MultiTaskDataset(self.train_files, is_train=False, valid_size=self.valid_size, is_molecular_dataset=self.imd, **self.kwargs)
        test = self.dataset_cls(self.test_files, is_test=True, **self.kwargs)

        train = DataLoader(train, batch_size=self.batch_size, collate_fn=collate,
                           sampler=WeightedRandomSampler(train.task_sizes, self.spe * self.batch_size), num_workers=self.nb_workers)
        valid = DataLoader(valid, batch_size=self.batch_size, collate_fn=collate,
                           num_workers=self.nb_workers)
        test = DataLoader(test, batch_size=1, collate_fn=collate)
        return train, valid, test
