# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Each solution to be tested should be stored in its own directory within
submissions/. The name of this new directory will serve as the ID for
the submission. If you wish to launch a RAMP challenge you will need to
provide an example solution within submissions/starting_kit/. Even if
you are not launching a RAMP challenge on RAMP Studio, it is useful to
have an example submission as it shows which files are required, how they
need to be named and how each file should be structured.
"""

import os
from collections import OrderedDict
from abc import ABCMeta
import progressbar
import numpy as np
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from dataset.openbhb import FeatureExtractor

############################################################################
# Define here your dataset
############################################################################

class Dataset(torch.utils.data.Dataset):
    """ A torch dataset for regression.
    """
    def __init__(self, X, y=None, transforms=None, indices=None):
        """ Init class.

        Parameters
        ----------
        X: array-like (n_samples, n_features)
            training data.
        y: array-like (n_samples, ), default None
            target values.
        transforms: list of callable, default None
            some transformations applied on each mini-batched input data.
        indices : array-like of shape (n_samples, ), default None
            the dataset indices. By default, the full dataset will be used.
        """
        self.transforms = transforms
        self.X = X
        self.y = y
        self.indices = indices
        if indices is None:
            self.indices = range(len(X))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_i = self.indices[i]
        X = self.X[real_i]
        for trf in self.transforms:
            X = trf.transform(X)
        X = torch.from_numpy(X)
        if self.y is not None:
            y = self.y[real_i]
            return X, y
        else:
            return X


class Standardizer(object):
    """ Standardize the input data.
    """
    def __init__(self, processes):
        self.processes = processes

    def fit(self, X, y):
        return self

    def transform(self, X):
        for process in self.processes:
            X = process(X)
        return X


class Normalize(object):
    """ Normalize the given n-dimensional array.
    """
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, X):
        _X = (
            self.std * (X - np.mean(X)) / (np.std(X) + self.eps) + self.mean)
        return _X


class Crop(object):
    """ Crop the given n-dimensional array either at a random location or
    centered.
    """
    def __init__(self, shape, type="center", keep_dim=False):
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type
        self.keep_dim = keep_dim

    def __call__(self, X):
        img_shape = np.array(X.shape)
        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.copping_type == "center":
                delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            elif self.copping_type == "random":
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(delta_before, delta_before + size[ndim]))
        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = X.copy()
            arr_copy[~mask] = 0
            return arr_copy
        _X = X[tuple(indexes)]
        return _X


class Pad(object):
    """ Pad the given n-dimensional array
    """
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs

    def __call__(self, X):
        _X = self._apply_padding(X)
        return _X

    def _apply_padding(self, arr):
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, **self.kwargs)
        return fill_arr


############################################################################
# Define here your model
############################################################################

class DenseNet(nn.Module):
    """3D-Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        mode (str) - "classifier" or "encoder" (all but last FC layer)
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, mode="classifier",
                 bn_size=4, num_classes=1000, in_channels=1,
                 memory_efficient=False):
        super(DenseNet, self).__init__()
        assert mode in ["classier", "encoder"], "Unknown mode: %s"%mode
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.mode = mode
        self.num_classes = num_classes
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features

        if mode == "classifier":
            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm3d(num_features))
            # Linear layer
            self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        if self.mode == "classifier":
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out.squeeze(dim=1)
        elif self.mode == "encoder":
            out = F.adaptive_avg_pool3d(features, 1)
            out = torch.flatten(out, 1)
            return out.squeeze(dim=1)
        else:
            raise ValueError("Unknown mode: %s"%self.mode)


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                memory_efficient=memory_efficient,
                )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


############################################################################
# Define here your regression model
############################################################################

class RegressionModel(metaclass=ABCMeta):
    """ Base class for Regression models.

    When the model has been trained locally, the trained weights are stored
    in the `__model_local_weights__` file.

    Some extra informations can be defined in the `__metadata_local_weights__`
    file. May be used to initialize some transformers without reaching
    some memory limitations by avoiding the fit on the train set.
    """
    __model_local_weights__ = os.path.join(
        os.path.dirname(__file__), "weights.pth")
    __metadata_local_weights__ = os.path.join(
        os.path.dirname(__file__), "metadata.pkl")

    def __init__(self, model, batch_size=15, transforms=None):
        """ Init class.

        Parameters
        ----------
        model: nn.Module
            the input model.
        batch_size:int, default 10
            the mini_batch size.
        transforms: list of callable, default None
            some transformations applied on each mini-batched input data.
        """
        self.model = model
        self.batch_size = batch_size
        self.transforms = transforms
        self.indices = None

    def fit(self, X, y):
        """ Restore weights.
        """
        self.model.train()
        if not os.path.isfile(self.__model_local_weights__):
            raise ValueError("You must provide the model weigths in your "
                             "submission folder.")
        state = torch.load(self.__model_local_weights__,
                           map_location="cpu")
        if "model" not in state:
            raise ValueError("Model weigths are searched in the state "
                             "dictionnary at the 'model' key location.")
        self.model.load_state_dict(state["model"], strict=False)

    def predict(self, X):
        """ Predict using the input model.

        Parameters
        ----------
        X: array-like (n_samples, n_features)
            samples.

        Returns
        -------
        outputs: array (n_samples, )
            returns predicted values.
        """
        self.model.eval()
        dataset = Dataset(X, transforms=self.transforms, indices=self.indices)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            outputs = []
            with progressbar.ProgressBar(max_value=len(testloader)) as bar:
                for cnt, inputs in enumerate(testloader):
                    inputs = inputs.float()
                    outputs.append(self.model(inputs))
                    bar.update(cnt)
            outputs = torch.cat(outputs, dim=0)
        return outputs.numpy()


############################################################################
# Define here your estimator pipeline
############################################################################

def get_estimator():
    """ Build your estimator here.

    Notes
    -----
    In order to minimize the memory load the first steps of the pipeline
    are applied directly as transforms attached to the Torch Dataset.

    Notes
    -----
    It is recommended to create an instance of sklearn.pipeline.Pipeline.
    """
    root = "'/neurospin/psy_sbox/bd261576/Datasets/OpenBHBChallenge/share/data_challenge'"
    net = DenseNet(32, (6, 12, 24, 16), 64, mode="encoder")
    selector = FeatureExtractor("vbm", root, sample_level=True)
    preproc = Standardizer([Crop((1, 121, 128, 121)),
                            Pad([1, 128, 128, 128], mode="constant"),
                            Normalize()])
    estimator = make_pipeline(
        RegressionModel(net, transforms=[selector, preproc]))
    return estimator
