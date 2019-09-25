import glob
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from scipy import sparse
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from collections import Counter

from tqdm import tqdm

from adkl.datasets import DATASETS_ROOT, episode_loader_molecules


def get_dataset_params(dataset_name):
    ds_folder = os.path.join(DATASETS_ROOT, dataset_name)

    DATASETS_MAPPING = dict(
        antibacterial=dict(
            files=[os.path.join(ds_folder, x) for x in glob.glob("{}/*.txt".format(ds_folder))],
            skip=0),
        bindingdb=dict(
            files=[os.path.join(ds_folder, x) for x in glob.glob("{}/*.tsv".format(ds_folder))],
            skip=2)
    )

    if dataset_name not in DATASETS_MAPPING:
        raise Exception(f"Unhandled dataset. The name of \
             the dataset should be one of those: {list(DATASETS_MAPPING.keys())}")

    filenames = DATASETS_MAPPING[dataset_name].get('files')
    return filenames, DATASETS_MAPPING[dataset_name].get('skip')


def get_fp_smiles(s):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, 2048)
    except:
        return None


def fps_to_csr_matrix(fps):
    n, m = len(fps), len(fps[0])
    idx, idy = zip(*[(i, j) for i, fp in enumerate(fps) for j, v in enumerate(fp) if v])
    x = sparse.csr_matrix((np.ones(len(idx)), (idx, idy)), shape=(n, m))
    return x

def set_pairwise_cross_similarity(mol_set1, mol_set2=None, verbose=False, n_pairs=None):
    if len(mol_set1) == 0:
       return 0
    start = time.time()
    fps1 =  Parallel(n_jobs=-1, verbose=verbose)(delayed(get_fp_smiles)(m) for m in mol_set1)
    fps1 = [fp for fp in fps1 if fp is not None]
    if mol_set2 is not None:
        fps2 = Parallel(n_jobs=-1, verbose=verbose)(delayed(get_fp_smiles)(m) for m in mol_set2)
        fps2 = [fp for fp in fps2 if fp is not None]
    else:
        fps2 = None

    x = fps_to_csr_matrix(fps1)
    if fps2 is not None:
        y = fps_to_csr_matrix(fps2)
    else:
        y = x
    print('csr', time.time() - start)

    n, m = x.shape[0], y.shape[0]
    if n_pairs is None or (n*m < n_pairs):
        inter = x.dot(y.transpose())
        s_x, s_y = np.array(x.sum(axis=1)).flatten(), np.array(y.sum(axis=1)).flatten()
        union = s_x[:, None] + s_y[None, :] - inter
        sim = inter / union
        sim = sim.mean()
    else:
        idx_x, idx_y = list(range(n)), list(range(m))
        i, j = random.choices(idx_x, k=n_pairs), random.choices(idx_y, k=n_pairs)
        x_i , x_j= x[i], y[j]
        inter = x_i.multiply(x_j).sum(axis=1)
        union = x_i.sum(axis=1) + x_j.sum(axis=1) - inter
        sim = np.array(inter / union).mean()
    return sim


def cluster_tasks_by_input_similarity():
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid", color_codes=True)

    data = dict()
    for dataset_name in ['bindingdb', 'antibacterial']:
        filenames, skip = get_dataset_params(dataset_name)
        n_tasks = len(filenames)
        sim_matrice = np.zeros((n_tasks, n_tasks))
        for i, fname in enumerate(filenames):
            for j in range(i+1):
                print(i, j)
                x_i, _, _, _ = episode_loader_molecules(filenames[i], skip=skip)
                x_j, _, _, _ = episode_loader_molecules(filenames[j], skip=skip)
                sim_matrice[i, j] = set_pairwise_cross_similarity(x_i, x_j, n_pairs=int(1e3))

        diag = np.diag(sim_matrice)
        sim_matrice = sim_matrice + sim_matrice.transpose()
        sim_matrice[np.diag_indices_from(sim_matrice)] = diag




if __name__ == '__main__':
    cluster_tasks_by_input_similarity()