from collections import OrderedDict

import numpy as np
import scipy.sparse as ss
import torch
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP, GetAvalonCountFP
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, \
    GetHashedTopologicalTorsionFingerprintAsBitVect, GetMACCSKeysFingerprint
from rdkit.Chem.rdmolops import RDKFingerprint, RenumberAtoms
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.base import TransformerMixin


def normalize_adj(adj):
    adj = adj + ss.eye(adj.shape[0])
    adj = ss.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ss.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()


def explicit_bit_vect_to_array(bitvector):
    """Convert a bit vector into an array

    Parameters
    ----------
    bitvector: rdkit.DataStructs.cDataStructs
        The struct of interest

    Returns
    -------
    res: np.ndarray
        array of binary elements
    """
    return np.array(list(map(int, bitvector.ToBitString())))


def one_of_k_encoding(val, allowed_choices):
    """Converts a single value to a one-hot vector.

    Parameters
    ----------
        val: class to be converted into a one hot vector
            (integers from 0 to num_classes).
        allowed_choices: a list of allowed choices for val to take

    Returns
    -------
        A list of size len(allowed_choices) + 1
    """
    encoding = np.zeros(len(allowed_choices) + 1, dtype=int)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(allowed_choices):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found
        encoding[-1] = 1
    return encoding


def totensor(x, gpu=True, dtype=torch.float):
    """convert a np array to tensor"""
    x = torch.from_numpy(x)
    x = x.type(dtype)
    if torch.cuda.is_available() and gpu:
        x = x.cuda()
    return x


def is_dtype_torch_tensor(dtype):
    r"""
    Verify if the dtype is a torch dtype

    Arguments
    ----------
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns
    -------
        A boolean saying if the dtype is a torch dtype
    """
    return isinstance(dtype, torch.dtype) or (dtype == torch.Tensor)


def is_dtype_numpy_array(dtype):
    r"""
    Verify if the dtype is a numpy dtype

    Arguments
    ----------
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns
    -------
        A boolean saying if the dtype is a numpy dtype
    """
    is_torch = is_dtype_torch_tensor(dtype)
    is_num = dtype in (int, float, complex)
    if hasattr(dtype, '__module__'):
        is_numpy = dtype.__module__ == 'numpy'
    else:
        is_numpy = False

    return (is_num or is_numpy) and not is_torch


class MoleculeTransformer(TransformerMixin):
    r"""
    Transform a molecule (rdkit.Chem.Mol object) into a feature representation.
    This class is an abstract class, and all its children are expected to implement the `_transform` method.
    """

    def __init__(self):
        super(MoleculeTransformer, self).__init__()

    def fit(self, X, y=None, **fit_params):
        return self

    @classmethod
    def to_mol(clc, mol, addHs=False, explicitOnly=True, ordered=True):
        r"""
        Convert an imput molecule (smiles representation) into a Chem.Mol
        :raises ValueError: if the input is neither a CHem.Mol nor a string

        .. CAUTION::
            As per rdkit recommandation, you need to be very careful about the molecules
            that Chem.AddHs outputs, since it is assumed that there is no hydrogen in the
            original molecule

        Arguments
        ----------
            mol: str or rdkit.Chem.Mol
                SMILES of a molecule or a molecule
            addHs: bool, optional): Whether hydrogens should be added the molecule.
               (Default value = False)
            explicitOnly: bool, optional
                Whether to only add explicit hydrogen or both
                (implicit and explicit) when addHs is set to True.
                (Default value = True)
            ordered: bool, optional, default=False
                Whether the atom should be ordered. This option is important if you want to ensure
                that the features returned will always maintain a sinfle atom order for the same molecule,
                regardless of its original smiles representation.

        Returns
        -------
            mol: rdkit.Chem.Molecule
                the molecule if some conversion have been made.
                If the conversion fails None is returned so make sure that you handle this case on your own.
        """
        if not isinstance(mol, (str, Chem.Mol)):
            raise ValueError("Input should be a CHem.Mol or a string")
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        # make more sense to add hydrogen before ordering
        if mol is not None and addHs:
            mol = Chem.AddHs(mol, explicitOnly=explicitOnly)
        if mol and ordered:
            new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
            mol = RenumberAtoms(mol, new_order)
        return mol

    def _transform(self, mol):
        r"""
        Compute features for a single molecule.
        This method need to be implemented by each child that inherits from MoleculeTransformer
        :raises NotImplementedError: if the method is not implemented by the child class
        Arguments
        ----------
            mol: Chem.Mol
                molecule to transform into features

        Returns
        -------
            features: the list of features

        """
        raise NotImplementedError('Missing implementation of _transform.')

    def transform(self, mols, ignore_errors=True, **kwargs):
        r"""
        Compute the features for a set of molecules.

        .. note::
            Note that depending on the `ignore_errors` argument, all failed
            featurization (caused whether by invalid smiles or error during
            data transformation) will be substitued by None features for the
            corresponding molecule. This is done, so you can find the positions
            of these molecules and filter them out according to your own logic.

        Arguments
        ----------
            mols: list(Chem.Mol) or list(str)
                a list containing smiles or Chem.Mol objects
            ignore_errors: bool, optional
                Whether to silently ignore errors
            kwargs:
                named arguments that are to be passed to the `to_mol` function.

        Returns
        --------
            features: a list of features for each molecule in the input set
        """

        features = []
        for i, mol in enumerate(mols):
            feat = None
            if ignore_errors:
                try:
                    mol = self.to_mol(mol, **kwargs)
                    feat = self._transform(mol)
                except:
                    pass
            else:
                mol = self.to_mol(mol, **kwargs)
                feat = self._transform(mol)
            features.append(feat)
        return features

    def __call__(self, mols, ignore_errors=True, **kwargs):
        r"""
        Calculate features for molecules. Using __call__, instead of transform. This function
        will force ignore_errors to be true, regardless of your original settings, and is offered
        mainly as a shortcut for data preprocessing. Note that most Transfomers allow you to specify
        a return datatype.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                SMILES of the molecules to be transformed
            ignore_errors: bool, optional
                Whether to ignore errors and silently fallback
                (Default value = True)
            kwargs: Named parameters for the transform method

        Returns
        -------
            feats: array
                list of valid features
            ids: array
                all valid molecule positions that did not failed during featurization

        See Also
        --------
            :func:`~ivbase.transformers.features.MoleculeTransformer.transform`

        """
        feats = self.transform(mols, ignore_errors=ignore_errors, **kwargs)
        ids = []
        for f_id, feat in enumerate(feats):
            if feat is not None:
                ids.append(f_id)
        return list(filter(None.__ne__, feats)), ids


class FingerprintsTransformer(MoleculeTransformer):
    r"""
    Fingerprint molecule transformer.
    This transformer is able to compute various fingerprints regularly used in QSAR modeling.

    Arguments
    ----------
        kind: str, optional
            Name of the fingerprinting method used. Should be one of
            {'global_properties', 'atom_pair', 'topological_torsion',
            'morgan_circular', 'estate', 'avalon_bit', 'avalon_count', 'erg',
            'rdkit', 'maccs'}
            (Default value = 'morgan_circular')
        length: int, optional
            Length of the fingerprint to use
            (Default value = 2000)

    Attributes
    ----------
        kind: str
            Name of the fingerprinting technique used
        length: int
            Length of the fingerprint to use
        fpfun: function
            function to call to compute the fingerprint
    """
    MAPPING = OrderedDict(
        # global_properties=lambda x, params: augmented_mol_properties(x),
        # physiochemical=lambda x: GetBPFingerprint(x),
        atom_pair=lambda x, params: GetHashedAtomPairFingerprintAsBitVect(
            x, **params),
        topological_torsion=lambda x, params: GetHashedTopologicalTorsionFingerprintAsBitVect(
            x, **params),
        ecfp2=lambda x, params: GetMorganFingerprintAsBitVect(
            x, 1, **params),
        ecfp4=lambda x, params: GetMorganFingerprintAsBitVect(
            x, 2, **params),
        ecfp6=lambda x, params: GetMorganFingerprintAsBitVect(
            x, 3, **params),
        estate=lambda x, params: FingerprintMol(x)[0],
        avalon_bit=lambda x, params: GetAvalonFP(x, **params),
        avalon_count=lambda x, params: GetAvalonCountFP(x, **params),
        erg=lambda x, params: GetErGFingerprint(x),
        rdkit=lambda x, params: RDKFingerprint(x, **params),
        maccs=lambda x, params: GetMACCSKeysFingerprint(x)
    )

    def __init__(self, kind='ecfp2', length=4096):
        super(FingerprintsTransformer, self).__init__()
        if not (isinstance(kind, str) and (kind in FingerprintsTransformer.MAPPING.keys())):
            raise ValueError("Argument kind must be in: " +
                             ', '.join(FingerprintsTransformer.MAPPING.keys()))
        self.kind = kind
        self.length = length
        self.fpfun = self.MAPPING.get(kind, None)
        if not self.fpfun:
            raise ValueError("Fingerprint {} is not offered".format(kind))
        self._params = {}
        self._params.update(
            {('fpSize' if kind == 'rdkit' else 'nBits'): length})

    def _transform(self, mol):
        r"""
        Transforms a molecule into a fingerprint vector
        :raises ValueError: when the input molecule is None

        Arguments
        ----------
            mol: rdkit.Chem.Mol
                Molecule of interest

        Returns
        -------
            fp: np.ndarray
                The computed fingerprint

        """

        if mol is None:
            raise ValueError("Expecting a Chem.Mol object, got None")
        # expect cryptic rdkit errors here if this fails, #rdkitdev
        fp = self.fpfun(mol, self._params)
        if isinstance(fp, ExplicitBitVect):
            fp = explicit_bit_vect_to_array(fp)
        else:
            fp = list(fp)
        return fp

    def transform(self, mols, **kwargs):
        r"""
        Transforms a batch of molecules into fingerprint vectors.

        .. note::
            The recommended way is to use the object as a callable.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                List of SMILES or molecules
            kwargs: named parameters for transform (see below)

        Returns
        -------
            fp: array
                computed fingerprints of size NxD, where D is the
                requested length of features and N is the number of input
                molecules that have been successfully featurized.

        See Also
        --------
            :func:`~ivbase.transformers.features.MoleculeTransformer.transform`

        """
        mol_list = [self.to_mol(mol, addHs=False) for i, mol in enumerate(mols)]
        # idx = [i for i, m in enumerate(mol_list) if m is None]
        mol_list = list(filter(None.__ne__, mol_list))
        features = np.array([self._transform(mol) for mol in mol_list]).astype(np.float32)
        features = totensor(features, gpu=False)

        return features

    def __call__(self, mols, dtype=torch.long, cuda=False, **kwargs):
        r"""
        Transforms a batch of molecules into fingerprint vectors,
        and return the transformation in the desired data type format as well as
        the set of valid indexes.

        Arguments
        ----------
            mols: (str or rdkit.Chem.Mol) iterable
                The list of input smiles or molecules
            dtype: torch.dtype or numpy.dtype, optional
                Datatype of the transformed variable.
                Expect a tensor if you provide a torch dtype, a numpy array if you provide a
                numpy dtype (supports valid strings) or a vanilla int/float. Any other option will
                return the output of the transform function.
                (Default value = torch.long)
            cuda: bool, optional
                Whether to transfer tensor on the GPU (if output is a tensor)
            kwargs: named parameters for transform (see below)

        Returns
        -------
            fp: array
                computed fingerprints (in `dtype` datatype) of size NxD,
                where D is the requested length of features and N is the number
                of input molecules that have been successfully featurized.
            ids: array
                all valid molecule positions that did not failed during featurization

        See Also
        --------
            :func:`~ivbase.transformers.features.FingerprintsTransformer.transform`

        """
        fp, ids = super(FingerprintsTransformer, self).__call__(mols, **kwargs)
        if is_dtype_numpy_array(dtype):
            fp = np.array(fp, dtype=dtype)
        elif is_dtype_torch_tensor(dtype):
            fp = totensor(fp, gpu=cuda, dtype=dtype)
        else:
            raise (TypeError('The type {} is not supported'.format(dtype)))
        return fp, ids
