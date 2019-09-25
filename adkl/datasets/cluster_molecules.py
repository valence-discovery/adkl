import os
import time
import hdbscan
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from adkl.datasets.molecule_toy import metadata_molecule_set, DATASETS_ROOT

FP_SIZE = 4096
USE_CACHE = False


def load_sets():
    if not USE_CACHE:
        antibacterial = set(metadata_molecule_set('antibacterial'))
        print('antibacterial size:', len(antibacterial))
        bindingdb = set(metadata_molecule_set('bindingdb')).difference(antibacterial)
        print('bindingdb size:', len(bindingdb))
        molecules_smiles = antibacterial.union(bindingdb)
        print('all molecules: ', len(molecules_smiles))

        # save all
        with open('ab_smiles.txt', 'w') as fd:
            fd.writelines([mol + '\n' for mol in antibacterial])
        with open('bdb_smiles.txt', 'w') as fd:
            fd.writelines([mol + '\n' for mol in bindingdb])
        with open('mol_smiles.txt', 'w') as fd:
            fd.writelines([mol + '\n' for mol in molecules_smiles])
    else:
        with open('ab_smiles.txt', 'r') as fd:
            antibacterial = fd.read().splitlines()
        print('antibacterial size:', len(antibacterial))
        with open('bdb_smiles.txt', 'r') as fd:
            bindingdb = fd.read().splitlines()
        print('bindingdb size:', len(bindingdb))
        with open('mol_smiles.txt', 'r') as fd:
            molecules_smiles = fd.read().splitlines()
        print('all molecules: ', len(molecules_smiles))
    return antibacterial, bindingdb, molecules_smiles



def get_fp_mol(x):
    return list(AllChem.GetMorganFingerprintAsBitVect(x, 3, FP_SIZE))


def get_fp_smiles(x):
    return get_fp_mol(Chem.MolFromSmiles(x))


def smiles_tranformer(smiles):
    x = Parallel(n_jobs=-1, verbose=1)(delayed(get_fp_smiles)(s) for s in smiles)
    return np.array(x)


def compute_molecular_features(reduced_set, full_set):
    if not USE_CACHE:
        ab_fps = smiles_tranformer(reduced_set)
        pca = PCA(n_components=100)
        pca.fit(ab_fps)
        print('PCA explained variance ratio :', pca.explained_variance_ratio_.sum())

        # transform all the molecules in vectors
        chunck_size = 100000
        molecules_smiles = list(full_set)
        arrs = np.concatenate([pca.transform(smiles_tranformer(molecules_smiles[i:i+chunck_size]))
                               for i in range(0, len(molecules_smiles), chunck_size)], axis=0)
        np.save('mol_features.npy', arrs)
    else:
        arrs = np.load('mol_features.npy')
    return arrs



def clustering(X):
    model = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=4)
    start = time.time()
    clusters = model.fit_predict(X)
    print(time.time() - start, clusters.max())
    print('nb clusters', clusters.max())
    return clusters


def save_clusters(mols, clusters):
    out_dir = os.path.join(DATASETS_ROOT, 'molecule_clusters')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'molecule_cluster_map.csv'), 'w') as fd:
        fd.writelines(["{},{}\n".format(mol, c) for mol, c in zip(mols, clusters)])



def show_clusters(mols, clusters):
    out_dir = os.path.join(DATASETS_ROOT, 'molecule_clusters')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clusters = [np.nonzero(clusters == i)[0] for i in range(clusters.max() + 1)]
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    clusters_sizes = [len(x) for x in clusters]
    print('NUMBER OF CLUSTERS :', len(clusters))
    print('NUMBER OF MOLECULES CLUSTERED :', sum(clusters_sizes))
    for cluster_idx in range(len(clusters)):
        size = len(clusters[cluster_idx])
        print('Cluster', cluster_idx, ':', size, 'molecules')
        # only show the first 8 molecule per cluster
        cluster = [Chem.MolFromSmiles(mols[idx]) for idx in clusters[cluster_idx][:20]]
        core = Chem.MolFromSmarts(rdFMCS.FindMCS(cluster).smartsString)
        fig = Draw.MolsToGridImage(cluster, molsPerRow=4, subImgSize=(200,200),
                                   highlightAtomLists=[m.GetSubstructMatch(core) for m in cluster])
        fig.save(os.path.join(out_dir, 'cluster_{}_size_{}.png'.format(cluster_idx, size)))



if __name__ == '__main__':
    antibacterial, bindingdb, molecules_smiles = load_sets()
    molecules_smiles =  list(molecules_smiles)
    X = compute_molecular_features(antibacterial, molecules_smiles)
    clusters = clustering(X)
    save_clusters(molecules_smiles, clusters)
    # show_clusters(molecules_smiles, clusters)



