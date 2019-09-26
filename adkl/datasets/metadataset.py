import warnings
from os.path import join, dirname, realpath

import numpy as np
import torch
from PIL import Image
from adkl.feature_extraction import FingerprintsTransformer
from adkl.feature_extraction.constants import AMINO_ACID_ALPHABET, SMILES_ALPHABET, ATOM_LIST
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASETS_ROOT = join(dirname(dirname(dirname(realpath(__file__)))), 'datasets')


class MetaRegressionDataset:
    INF = int(1e9)

    def __init__(self, tasks_filenames, episode_loader, x_transformer=None,
                 y_transformer=None, task_descriptor_transformer=None,
                 raw_inputs=False, max_examples_per_episode=None,
                 max_test_examples=None, is_test=False, seed=42,
                 nb_query_samples=0, y_scaler=None,
                 nb_test_time_replicates=20, **kwargs):
        super(MetaRegressionDataset, self).__init__()
        assert nb_query_samples >= 0, "nb_query_samples must be non negative"
        self.tasks_filenames = tasks_filenames[:]
        self.max_examples_per_episode = max_examples_per_episode
        self.max_test_examples = (200 if max_test_examples is None else max_test_examples)
        self.rgn = np.random.RandomState(seed)
        self.is_test = is_test
        self.episode_loader = episode_loader
        self.nb_query_samples = nb_query_samples
        self.y_scaler = y_scaler
        self.nb_test_time_replicates = nb_test_time_replicates

        if x_transformer is None:
            self.x_transformer = lambda x: torch.FloatTensor(x)
        else:
            self.x_transformer = x_transformer
        if y_transformer is None:
            self.y_transformer = lambda y: torch.FloatTensor(y)
        else:
            self.y_transformer = y_transformer
        if task_descriptor_transformer is None:
            self.task_descriptor_transformer = lambda task_descriptor: torch.FloatTensor(task_descriptor)
        else:
            self.task_descriptor_transformer = task_descriptor_transformer
        self.raw_inputs = raw_inputs
        self.init_tasks_sizes()

    def __len__(self):
        return (self.nb_test_time_replicates * len(self.tasks_filenames)) if self.is_test else self.INF

    def init_tasks_sizes(self):
        def file_len(fname):
            if fname.startswith('__'):
                return int(fname.split('__')[1])

            elif fname.endswith('.dat'):
                return len(np.memmap(fname, dtype='float32', mode='r'))

            with open(fname) as f:
                for i, _ in enumerate(f):
                    pass

            return i + 1

        task_sizes = np.array([file_len(f) for f in self.tasks_filenames])
        s = np.log2(task_sizes)
        self.task_weights = s / np.sum(s)

    def count_unique_inputs(self):
        temp = np.concatenate([self.episode_loader(f)[0]
                               for f in self.tasks_filenames])
        return len(set(temp))

    def _episode(self, xtrain, ytrain, xtest, ytest, xquery=None, yquery=None, task_descriptor=None, idx=None):
        if self.raw_inputs:
            return dict(Dtrain=(xtrain, ytrain),
                        Dtest=(xtest, ytest),
                        idx=idx,
                        tasks_descr=task_descriptor), ytest

        others = dict()

        if xquery is None or yquery is None:
            ntrain = len(xtrain)
            temp = self.x_transformer(np.concatenate([xtrain, xtest]))
            xtrain, xtest = temp[:ntrain], temp[ntrain:]
            temp = self.y_transformer(np.concatenate([ytrain, ytest]))
            ytrain, ytest = temp[:ntrain], temp[ntrain:]

            query = dict()

        else:
            ntrain = len(xtrain)
            ntest = len(xtest)

            temp = self.x_transformer(np.concatenate([xtrain, xtest, xquery]))
            xtrain, xtest, xquery = temp[:ntrain], temp[ntrain:ntrain + ntest], temp[ntrain + ntest:]
            temp = self.y_transformer(np.concatenate([ytrain, ytest, yquery]))
            ytrain, ytest, yquery = temp[:ntrain], temp[ntrain:ntrain + ntest], temp[ntrain + ntest:]

            query = dict(Dquery=(self.cuda_tensor(xquery), self.cuda_tensor(yquery)))

        if task_descriptor is not None:
            task_descriptor = self.task_descriptor_transformer(task_descriptor)
            td = dict(task_descr=self.cuda_tensor(task_descriptor))
        else:
            td = dict()

        others.update(query)
        others.update(td)
        return (dict(Dtrain=(self.cuda_tensor(xtrain), self.cuda_tensor(ytrain)),
                     Dtest=(self.cuda_tensor(xtest), self.cuda_tensor(ytest)),
                     idx=idx,
                     **others),
                self.cuda_tensor(ytest))

    @staticmethod
    def cuda_tensor(x, use_available_gpu=True):
        if isinstance(x, torch.Tensor) and torch.cuda.is_available() and use_available_gpu:
            x = x.cuda()
        return x

    def __getitem__(self, i):
        file_index = i % len(self.tasks_filenames)
        filename = self.tasks_filenames[file_index]
        x, y, task_descriptor, scaler = self.episode_loader(filename, self.y_scaler)
        n, indexes = len(x), np.arange(len(x))

        self.rgn.shuffle(indexes)

        if self.is_test:
            if self.max_examples_per_episode < 1:
                train_indexes, test_indexes = train_test_split(indexes, train_size=self.max_examples_per_episode)
            else:
                k = min(int(n / 2), self.max_examples_per_episode)
                train_indexes, test_indexes = indexes[:k], indexes[k:k + self.max_test_examples]
                remaining_indexes = indexes[k + self.max_test_examples:]
        else:
            n = min(2 * self.max_examples_per_episode, n)
            temp_indexes = indexes[:n]
            k = int(n / 2)
            train_indexes, test_indexes = temp_indexes[:k], temp_indexes[k:(2 * k)]
            remaining_indexes = indexes[2 * k:]
        y_train = y[train_indexes]
        std_zero = np.std(y[train_indexes]) == 0
        if std_zero:
            y_train = np.array(y_train) + np.random.normal(0, np.abs(np.mean(y_train) / 10.0) + 1e-8)

        if self.nb_query_samples > 0:
            if len(remaining_indexes) < self.nb_query_samples:
                warnings.warn("Some episodes don't have enough sample in the query set. \n" +
                              "You may want to change the train_size or the test_size")
            query_indexes = remaining_indexes[:self.nb_query_samples]
            xquery, yquery = x[query_indexes], y[query_indexes]
        else:
            xquery, yquery = None, None

        xtrain = x[train_indexes]
        ytrain = y[train_indexes].reshape(-1, 1)
        xtest = x[test_indexes]
        ytest = y[test_indexes].reshape(-1, 1)

        if scaler is None:
            def scaler(v):
                return v

        ytrain = scaler(ytrain)
        ytest = scaler(ytest)

        if yquery is not None:
            yquery = yquery.reshape(-1, 1)
            yquery = scaler(yquery)

        return self._episode(xtrain=xtrain, ytrain=ytrain,
                             xtest=xtest, ytest=ytest,
                             xquery=xquery, yquery=yquery,
                             task_descriptor=task_descriptor, idx=file_index)


class MoleculeMetaDataset(MetaRegressionDataset):
    def __init__(self, *args, representation: str = 'smiles', length: int = 4096, **kwargs):
        super(MoleculeMetaDataset, self).__init__(*args, **kwargs)
        if kwargs.get('nb_query_samples', 0) > 0:
            raise Exception("Active learning mode not supported for this dataset yet")

        self.representation = representation.lower()

        if self.representation.startswith('ecfp'):
            transformer = FingerprintsTransformer(kind=self.representation, length=length)
        else:
            raise AttributeError(f'{self.representation} is not implemented')

        self.x_transformer = lambda x: transformer.transform(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descr_transformer = lambda z: None
        self.max_test_examples = 1000


class MetaDataLoader(object):

    def __init__(self, train_files, test_files, batch_size=32, steps_per_epoch=1000, valid_size=0.25,
                 is_molecule_dataset=False, **kwargs):
        # print(kwargs)
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.kwargs = kwargs
        self.nb_workers = 0  # os.cpu_count()
        self.train_files = train_files
        self.test_files = test_files
        self.spe = steps_per_epoch

        self.dataset_cls = MoleculeMetaDataset if is_molecule_dataset else MetaRegressionDataset

    def load_partitions(self):
        def collate(x):
            return list(zip(*x))

        train_files, valid_files = train_test_split(self.train_files, test_size=self.valid_size)
        train = self.dataset_cls(train_files, **self.kwargs)
        valid = self.dataset_cls(valid_files, **self.kwargs)
        test = self.dataset_cls(self.test_files, is_test=True, **self.kwargs)

        train = DataLoader(train, collate_fn=collate, batch_size=self.batch_size,
                           sampler=WeightedRandomSampler(train.task_weights, self.spe * self.batch_size),
                           num_workers=self.nb_workers)
        valid = DataLoader(valid, batch_size=self.batch_size, collate_fn=collate,
                           num_workers=self.nb_workers)
        test = DataLoader(test, batch_size=1, collate_fn=collate)
        return train, valid, test
