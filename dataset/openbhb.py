from torch.utils.data.dataset import Dataset
from nilearn.masking import unmask
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import os, re
import bisect
import nibabel
from collections import OrderedDict
import pandas as pd
import numpy as np
from typing import Callable, Any, Sequence, Type

class OpenBHB(Dataset):
    """
        OpenBHB is a torch.Dataset written in a torchvision-like manner. It is memory-efficient, with lazy loading.
        It comes with a train/test split (stratified on age/sex/site) that is used for local RAMP submission
        (cf https://paris-saclay-cds.github.io/ramp-workflow/command_line.html).
        Data must be downloaded here: https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing
        with a (free) IEEE account.

        It contains:
        ... 3 different pre-processings:
            - Quasi-Raw (Minimal)
            - VBM (CAT12)
            - SBM (FreeSurfer)
        ... target to predict: 'age', 'site', or both 'age+site'
        ... 3-fold CV scheme with (train, internal test, external test) for each fold.
        ... several meta-data:
            - unique identifier across pre-processing and split (participant_id)
            - TIV + Global CSF/WM/GM volumes + acq. settings
    Attributes:
          * samples: list of (path, target)
          * shape, tuple: shape of the data
          * infos: pd DataFrame: TIV + global CSF/WM/GM + acq. settings and magnetic field strength, "sex", "age", "site"
          * metadata: ROI names or VBM/quasi-raw template used
    """
    def __init__(self, root: str, preproc: str='vbm', split: str='train', target: str='age+site',
                 cv: int=None, transforms: Callable[[np.ndarray], np.ndarray]=None,
                 target_transforms: Callable[[int, float], Any]=None):
        """
        :param root: str, path to the root directory containing the downloaded zip files.
        :param preproc: str, must be either VBM ('vbm'), ROI-VBM ('vbm_roi'), Quasi-Raw ('quasi_raw'),
                     FSL-Xhemi ('fsl_xhemi'), FSL Desikan ROI ('fsl_desikan_roi') or
                     FSL Destrieux ROI ('fsl_destrieux_roi')
        :param split: str, either 'train+test' (train+test) 'train' (train), 'test'
                     (stratified test) or 'external_test' (external test with no site overlap)
        :param target: str, either 'age+site' (as list [float, int]), 'age' (float) or 'site' (int)
        :param cv: int, if not None, loads the fold required with the CV scheme proposed
        :param transforms (callable, optional): A function/transform that takes a single input data
               (e.g a 3D image for VBM) and returns a transformed version.
        :param target_transforms (callable, optional): A function/transform that takes a target
               and returns a transformed version.
        """

        assert preproc in ['vbm', 'vbm_roi', 'quasi_raw', 'fsl_xhemi', 'fsl_desikan_roi', 'fsl_destrieux_roi'], \
            "Unknown preproc: %s"%preproc
        assert split in ['train+test', 'train', 'test', 'external_test'], "Unknown split: %s"%split
        assert target in ['age', 'site', 'age+site'], "Unknown target: %s"%target
        if cv is not None:
            assert cv in [0, 1, 2], "Incorrect fold selected (must be in {0, 1, 2}): %i"%cv
        self.root = root
        self.preproc = preproc
        self.split = split
        self.cv = cv
        self.target_name = ["age", "site"]
        self.transforms = transforms
        self.target_transforms = target_transforms

        if not self._check_integrity():
            raise RuntimeError("Files not found. Check the the root directory %s"%root)

        self.extractor = FeatureExtractor(dtype=preproc, root=root, sample_level=True)

        # Load data with memory-mapping
        select = target
        if target == 'age+site':
            select = ["age", "site"]
        if cv is None:
            if self.split == "train+test":
                all_labels = pd.concat((pd.read_csv(os.path.join(self.root, "data", "train.tsv"), sep="\t"),
                                        pd.read_csv(os.path.join(self.root, "data", "test.tsv"), sep="\t"),
                                        pd.read_csv(os.path.join(self.root, "data", "external_test.tsv"), sep="\t")),
                                       sort=False)
                self.samples = LazyConcat([np.load(os.path.join(self.root, "data", "%s.npy" % split), mmap_mode="r")
                                           for split in ["train", "test", "external_test"]])
                self.targets = all_labels[select].copy().values
            elif self.split in ["train", "test", "external_test"]:

                all_labels = pd.read_csv(os.path.join(self.root, "data", "%s.tsv"%self.split), sep="\t")
                self.samples = np.load(os.path.join(self.root, "data", "%s.npy"%self.split), mmap_mode="r")
                self.targets = all_labels[select].copy().values
            else:
                raise ValueError("Unknown split: %s"%self.split)

        else: # CV-mode
            if self.split == "train+test":
                raise ValueError("Unset CV-scheme when selecting 'train+test' split")
            # Get the corresponding train/internal test/external test
            try:
                scheme = pd.read_json(os.path.join(self.root, "data", "cv_splits.json"))["fold%i"%cv]
            except Exception as e:
                raise ValueError("Unexpected error when loading CV-scheme. Check %s" %
                                 os.path.join(self.root, "data", "cv_splits.json"))
            all_labels = pd.read_csv(os.path.join(self.root, "data", "train.tsv"), sep="\t")
            split = "internal_test" if self.split == "test" else self.split
            mask = all_labels["participant_id"].isin(scheme[split]).values.astype(np.bool)
            all_labels = all_labels[mask]
            self.samples = LazyMaskedArray(data=np.load(os.path.join(self.root, "data", "train.npy"),
                                            mmap_mode="r"), bool_mask=mask)
            self.targets = all_labels[select].copy().values

        # Get meta-data associated to specific preproc (ROI names, or template used)
        self.metadata = self._extract_metadata()
        self.infos = all_labels.copy()
        self.shape = (len(self.samples), *self[0][0].shape)

    def _check_integrity(self):
        """
        Check the integrity of root dir (including the directories/files required). It does NOT check their content.
        Should be formatted (when compressed) as :
        /root
            -----RAMP-CHALLENGE-----
            /challenge_data.zip
                /data
                    train.npy
                    train.tsv
                    test.npy
                    test.tsv
                    external_test.npy
                    external_test.tsv
                    cv_splits.json
            -----COMMON FILES-----
                /resource
                    cat12vbm_labels.txt
                    freesurfer_atlas-desikan_labels.txt
                    freesurfer_atlas-destrieux_labels.txt
                    freesurfer_channels.txt
                    freesurfer_xhemi_channels.txt
                    cat12vbm_space-MNI152_desc-gm_TPM.nii.gz
                    quasiraw_space-MNI152_desc-brain_T1w.nii.gz
        """
        self._extract_archives("challenge_data.zip", remove_finished=True)

        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "train.npy"))
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "train.tsv"))
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "test.npy"))
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "test.tsv"))
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "external_test.npy"))
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "external_test.tsv"))
        is_complete &= os.path.isfile(os.path.join(self.root, "data", "cv_splits.json"))

        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "cat12vbm_labels.txt"))
        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "freesurfer_atlas-desikan_labels.txt"))
        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "freesurfer_atlas-destrieux_labels.txt"))
        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "freesurfer_channels.txt"))
        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "freesurfer_xhemi_channels.txt"))
        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"))
        is_complete &= os.path.isfile(os.path.join(self.root, "resource", "quasiraw_space-MNI152_desc-brain_T1w.nii.gz"))

        return is_complete

    def _extract_archives(self, folders, remove_finished=False):
        if isinstance(folders, str):
            folders = [folders]
        for f in folders:
            pth_zip = os.path.join(self.root, f)
            if os.path.isfile(pth_zip):
                import zipfile
                with zipfile.ZipFile(pth_zip, 'r') as zip_file:
                    zip_file.extractall(path=self.root)
                if remove_finished:
                    os.remove(pth_zip)

    def _extract_metadata(self):
        """
        :return: ROI names or VBM/Quasi-Raw templates
        """
        if self.preproc == "vbm_roi":
            meta = pd.read_csv(os.path.join(self.root, "resource", "cat12vbm_labels.txt"), names=["ROI"])
        elif self.preproc == "vbm":
            meta = nibabel.load(os.path.join(self.root, "resource", "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"))
        elif self.preproc == "quasi_raw":
            meta = nibabel.load(os.path.join(self.root, "resource", "quasiraw_space-MNI152_desc-brain_T1w.nii.gz"))
        elif self.preproc == "fsl_xhemi":
            meta = pd.read_csv(os.path.join(self.root, "resource", "freesurfer_xhemi_channels.txt"))
        elif self.preproc == "fsl_desikan_roi":
            meta = pd.read_csv(os.path.join(self.root, "resource", "freesurfer_atlas-desikan_labels.txt"), names=["ROI"])
        elif self.preproc == "fsl_destrieux_roi":
            meta = pd.read_csv(os.path.join(self.root, "resource", "freesurfer_atlas-destrieux_labels.txt"), names=["ROI"])
        else:
            raise ValueError("Unknown preproc: %s"%self.preproc)
        return meta

    def load_sample(self, idx):
        return self.extractor.transform(self.samples[idx])

    def get_infos(self):
        return self.infos

    def get_metadata(self):
        return self.metadata

    @staticmethod
    def get_mean_var(root, preproc):
        """
            Get mean/var over training set for this preproc.
        """
        # Get all training data as big matrix X [n x *] where n == # subjects and * == features dim (e.g 4D for img)
        train_data = OpenBHB(root, preproc, split="train").get_data()[0]
        return np.nanmean(train_data, axis=0), np.nanstd(train_data, axis=0)

    def get_data(self, indices: Sequence[int]=None, mask: np.ndarray=None, dtype: Type=np.float32):
        """
        Loads all (or selected ones) data in memory and returns a big numpy array X_data with y_data
        The input/target transforms are ignored.
        Warning: this can be VERY memory-consuming (~40GB if all data are loaded)
        :param indices (Optional): list of indices to load
        :param mask (Optional binary mask): binary mask to apply to the data. Each 3D volume is transformed into a
        vector. Can be 3D mask or 4D (channel + img)
        :param dtype (Optional): the final type of data returned (e.g np.float32)
        :return (np.ndarray, np.ndarray), a tuple (X, y)
        """
        (tf, target_tf) = (self.transforms, self.target_transforms)
        self.transforms, self.target_transforms = None, None
        targets = []
        if mask is not None:
            assert len(mask.shape) in [3, 4], "Mask must be 3D or 4D (current shape is {})".format(mask.shape)
            if len(mask.shape) == 3:
                # adds the channel dimension
                mask = mask[np.newaxis, :]
        if indices is None:
            nbytes = np.product(self.shape) if mask is None else mask.sum() * len(self)
            print("Dataset size to load (shape {}): {:.2f} GB".format(self.shape, nbytes*np.dtype(dtype).itemsize/
                                                                      (1024*1024*1024)), flush=True)
            if mask is None:
                data = np.zeros(self.shape, dtype=dtype)
            else:
                data = np.zeros((len(self), mask.sum()), dtype=dtype)
            for i in range(len(self)):
                (sample, target) = self[i]
                data[i] = sample[mask] if mask is not None else sample
                targets.append(target)
            self.transforms, self.target_transforms = (tf, target_tf)
            return data, np.array(targets)
        else:
            nbytes = np.product(self.shape[1:]) * len(indices) if mask is None else mask.sum() * len(indices)
            print("Dataset size to load (shape {}): {:.2f} GB".format((len(indices),) + self.shape[1:],
                                                                      nbytes*np.dtype(dtype).itemsize/
                                                                      (1024*1024*1024)), flush=True)
            if mask is None:
                data = np.zeros((len(indices), *self.shape[1:]), dtype=dtype)
            else:
                data = np.zeros((len(indices), mask.sum()), dtype=dtype)
            for i, idx in enumerate(indices):
                (sample, target) = self[i]
                data[i] = sample[mask] if mask is not None else sample
                targets.append(target)
            self.transforms, self.target_transforms = (tf, target_tf)
            return data.astype(dtype), np.array(targets)

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            raise TypeError("Expected <int> idx (got {})".format(type(idx)))
        sample = self.load_sample(idx)
        target = self.targets[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return "OpenBHB-%s-%s"%(self.preproc, self.split)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the requested data associated features from the input buffered data.
    """
    def __init__(self, dtype, root, sample_level=False):
        """ Init class.
        Parameters
        ----------
        dtype: str
            the requested data: 'vbm', 'quasiraw', 'fsl_xhemi', 'vbm_roi', 'desikan_roi',
            or 'destrieux_roi'.
        root: str
            path to root directory where OpenBHB data were downloaded
        sample_level: bool
            Whether to transform individual sample with shape (n_features,)
            or array of samples with shape (n_samples, n_features)
        """
        self.MODALITIES = OrderedDict([
            ("vbm", {
                "shape": (-1, 1, 121, 145, 121),
                "size": 519945}),
            ("quasi_raw", {
                "shape": (-1, 1, 182, 218, 182),
                "size": 1827095}),
            ("fsl_xhemi", {
                "shape": (-1, 7, 187248),
                "size": 1310736}),
            ("vbm_roi", {
                "shape": (-1, 1, 284),
                "size": 284}),
            ("fsl_desikan_roi", {
                "shape": (-1, 7, 68),
                "size": 476}),
            ("fsl_destrieux_roi", {
                "shape": (-1, 7, 148),
                "size": 1036})])
        self.MASKS = {
            "vbm": {
                "path": os.path.join(root, "resource", "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"),
                "thr": 0.05},
            "quasi_raw": {
                "path": os.path.join(root, "resource", "quasiraw_space-MNI152_desc-brain_T1w.nii.gz"),
                "thr": 0}
        }

        if dtype not in self.MODALITIES:
            raise ValueError("Invalid input data type.")
        self.sample_level = sample_level
        self.dtype = dtype
        data_types = list(self.MODALITIES.keys())
        index = data_types.index(dtype)
        cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        if index > 0:
            self.start = cumsum[index - 1]
        else:
            self.start = 0
        self.stop = cumsum[index]
        self.masks = dict((key, val["path"]) for key, val in self.MASKS.items())
        for key in self.masks:
            arr = nibabel.load(self.masks[key]).get_fdata()
            thr = self.MASKS[key]["thr"]
            arr[arr <= thr] = 0
            arr[arr > thr] = 1
            self.masks[key] = nibabel.Nifti1Image(arr.astype(int), np.eye(4))

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.sample_level:
            select_X = X[self.start:self.stop]
        else:
            select_X = X[:, self.start:self.stop]
        if self.dtype in ("vbm", "quasi_raw"):
            im = unmask(select_X, self.masks[self.dtype])
            select_X = im.get_fdata()
        if self.sample_level:
            select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"][1:])
        else:
            select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"])
        return select_X


class LazyConcat(object):
    """
        Handles concatenation of NumPy arrays in a lazy manner.
        If deals with common indexation (int index and slicer)
    """
    def __init__(self, arrays):
        self.arrays = arrays
        self.cum_dims = np.cumsum([len(arr) for arr in self.arrays])
        if len(self.arrays) > 0:
            assert np.all([isinstance(arr, np.ndarray) for arr in self.arrays]), "All arrays must by NumPy array"
            assert np.all([np.all(arrays[i].shape[1:] == arrays[0].shape[1:])
                           for i in range(len(arrays))]), "All shapes must be equal except on first dimension"
            self.shape = (len(self), *self.arrays[0].shape[1:])
        else:
            self.shape = ()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return np.concatenate([self[i] for i in range(*item.indices(len(self)))])
        elif isinstance(item, int):
            i = bisect.bisect_right(self.cum_dims, item)
            j = (item - self.cum_dims[i-1]) if i > 0 else item
            return self.arrays[i][j]
        else:
            raise TypeError("Invalid argument type")

    def __len__(self):
        if len(self.cum_dims) == 0:
            return 0
        return self.cum_dims[-1]

class LazyMaskedArray(object):
    """
    Performs lazy masking with a binary mask. It handles common indexation (int index and slicer).
    """
    def __init__(self, data, bool_mask):
        self.data = data
        assert isinstance(bool_mask, np.ndarray) and bool_mask.dtype == np.bool
        assert len(self.data) == len(bool_mask)
        self.mask = np.arange(len(self.data))[bool_mask].astype(np.int64)
        self.dtype = data.dtype

    def __getitem__(self, item):
        if isinstance(item, slice):
            return np.concatenate([self[i] for i in range(*item.indices(len(self)))])
        elif isinstance(item, int):
            return self.data[self.mask[item]]
        else:
            raise TypeError("Invalid argument type")

    def __len__(self):
        return len(self.mask)
