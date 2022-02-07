# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
This parametrizes the setup using building blocks from RAMP workflow.
"""

import os
import pandas as pd
import numpy as np
import rampwf as rw
from rampwf.utils.pretty_print import print_title
from sklearn.base import BaseEstimator
from sklearn.utils import _safe_indexing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, r2_score, balanced_accuracy_score, accuracy_score)


class DeepDebiasingEstimator(rw.workflows.SKLearnPipeline):
    """ Wrapper to convert a scikit-learn estimator into a Deep Learning
    Debiasing RAMP workflow that use a public and a private dataset as well
    as two regressors (age & site) on latent space network features.

    Notes
    -----
    The training is not performed on the server side. Weights are required at
    the submission time and fold indices are tracked at training time.
    """
    def __init__(self, filename="estimator.py", additional_filenames=None,
                 max_n_features=10000):
        super().__init__(filename, additional_filenames)
        self.max_n_features = max_n_features

    def train_submission(self, module_path, X, y, train_idx=None):
        """ Train the estimator of a given submission.

        Parameters
        ----------
        module_path : str
            The path to the submission where `filename` is located.
        X : array-like, dataframe of shape (n_samples, n_features)
            The data matrix.
        y : array-like of shape (n_samples, 2)
            The target vector: age, site.
        train_idx : array-like of shape (n_training_samples,), default None
            The training indices. By default, the full dataset will be used
            to train the model. If an array is provided, `X` and `y` will be
            subsampled using these indices.

        Returns
        -------
        estimators : estimator objects
            Scikit-learn models fitted on (`X`, `y`).
        """
        print_title("DeepDebiasingEstimator...")
        _train_idx = (slice(None, None, None) if train_idx is None
                      else train_idx)
        submission_module = rw.utils.importing.import_module_from_source(
            os.path.join(module_path, self.filename),
            os.path.splitext(self.filename)[0],
            sanitize=True
        )
        os.environ["VBM_MASK"] = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz")
        os.environ["QUASIRAW_MASK"] = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "quasiraw_space-MNI152_desc-brain_T1w.nii.gz")
        estimator = submission_module.get_estimator()
        y_train = _safe_indexing(y, _train_idx)
        y_age, y_site = y_train.T
        y_age = y_age.astype(float)
        y_site = y_site.astype(int)
        print("- X:", X.shape, len(_train_idx))
        print("- y:", y_train.shape)
        print("- ages:", y_age.shape)
        print("- sites:", y_site.shape)
        print_title("Restoring weights...")
        features_estimator = estimator.fit(None, None)
        print_title("Estimates features...")
        for item in estimator:
            if hasattr(item, "indices"):
                item.indices = train_idx
        features = features_estimator.predict(X)
        for item in estimator:
            if hasattr(item, "indices"):
                item.indices = None
        if features.shape[1] > self.max_n_features:
            raise ValueError(
                "You reached the maximum authorized size of the feature space "
                "({0}).".format(self.max_n_features))
        print("- features:", features.shape)
        print_title("Estimates sites from features...")
        site_estimator = SiteEstimator()
        site_estimator.fit(features, y_site)
        print_title("Estimates age from features...")
        age_estimator = AgeEstimator()
        age_estimator.fit(features, y_age)
        return features_estimator, site_estimator, age_estimator

    def test_submission(self, estimators_fitted, X):
        """Predict using a fitted estimator.

        Parameters
        ----------
        estimators_fitted : estimator objects
            Fitted scikit-learn estimators.
        X : array-like, dataframe of shape (n_samples, n_features)
            The test data set.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes) or (n_samples)
        """
        print_title("Testing...")
        print("- X:", X.shape)
        features_estimator, site_estimator, age_estimator = estimators_fitted
        for item in features_estimator:
            if hasattr(item, "indices"):
                item.indices = range(1, len(X))
        features = features_estimator.predict(X)
        split_idx = X[0, 0]
        internal_features, external_features = split_data(features, split_idx)
        y_site_pred = site_estimator.predict(internal_features)
        y_age_pred = age_estimator.predict(internal_features)
        print("- internal features:", internal_features.shape)
        print("- y site [internal]:", y_site_pred.shape)
        print("- y age [internal]:", y_age_pred.shape)
        if len(external_features) > 0:
            y_age_external_pred = age_estimator.predict(external_features)
            print("- features [external]:", external_features.shape)
            print("- y age [external]:", y_age_external_pred.shape)
            y_age_pred = np.concatenate(
                (y_age_pred, y_age_external_pred), axis=0)
            null_pred = np.empty(
                (len(y_age_external_pred), y_site_pred.shape[1]))
            null_pred[:] = np.nan
            y_site_pred = np.concatenate((y_site_pred, null_pred), axis=0)
        y_age_pred = np.concatenate(([np.nan], y_age_pred), axis=0)
        header = np.empty((1, y_site_pred.shape[1]))
        header[:] = np.nan
        y_site_pred = np.concatenate((header, y_site_pred), axis=0)
        print("- y site [fusion]:", y_site_pred.shape)
        print("- y age [fusion]:", y_age_pred.shape)
        return np.concatenate([y_age_pred.reshape(-1, 1), y_site_pred], axis=1)


def split_data(arr, split_idx):
    """ Split the data.
    """
    split_idx = int(split_idx)
    return arr[:split_idx], arr[split_idx:]


class SiteEstimator(BaseEstimator):
    """ Define the site estimator on latent space network features.
    """
    def __init__(self):
        self.site_estimator = GridSearchCV(
            LogisticRegression(solver="saga", max_iter=150), cv=5,
            param_grid={"C": 10.**np.arange(-2, 3)},
            scoring="balanced_accuracy")

    def fit(self, X, y):
        self.site_estimator.fit(X, y)

    def predict(self, X):
        y_pred = self.site_estimator.predict_proba(X)
        return y_pred


class AgeEstimator(BaseEstimator):
    """ Define the age estimator on latent space network features.
    """
    def __init__(self):
        self.age_estimator = GridSearchCV(
            Ridge(), param_grid={"alpha": 10.**np.arange(-2, 3)}, cv=5,
            scoring="r2")

    def fit(self, X, y):
        self.age_estimator.fit(X, y)

    def predict(self, X):
        y_pred = self.age_estimator.predict(X)
        return y_pred


class R2(rw.score_types.BaseScoreType):
    """ Compute coefficient of determination usually denoted as R2.
    """
    is_lower_the_better = False
    minimum = -float("inf")
    maximum = 1

    def __init__(self, name="mae", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        is_test_set = np.isnan(y_pred[0])
        if is_test_set:
            split_idx = y_true[0]
            y_pred, _ = split_data(y_pred[1:], split_idx)
            y_true, _ = split_data(y_true[1:], split_idx)
        return r2_score(y_true, y_pred)


class MAE(rw.score_types.BaseScoreType):
    """ Compute mean absolute error, a risk metric corresponding to the
    expected value of the absolute error loss or l1-norm loss.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="mae", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        is_test_set = np.isnan(y_pred[0])
        if is_test_set:
            split_idx = y_true[0, 0]
            y_pred, _ = split_data(y_pred[1:], split_idx)
            y_true, _ = split_data(y_true[1:], split_idx)
        return mean_absolute_error(y_true, y_pred)


class ExtMAE(rw.score_types.BaseScoreType):
    """ Compute mean absolute error on the private external test set.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="mae", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        is_test_set = np.isnan(y_pred[0])
        if is_test_set:
            split_idx = y_true[0, 0]
            y_pred, y_pred_external = split_data(y_pred[1:], split_idx)
            y_true, y_true_external = split_data(y_true[1:], split_idx)
        else:
            y_pred_external = y_pred
            y_true_external = y_true
        return mean_absolute_error(y_true_external, y_pred_external)


class BACC(rw.score_types.classifier_base.ClassifierBaseScoreType):
    """ Compute balanced accuracy, which avoids inflated performance
    estimates on imbalanced datasets.
    """
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="bacc", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        is_test_set = (-1 in y_pred_label_index)
        if is_test_set:
            split_idx = y_true_label_index[0]
            y_true_label_index, _ = split_data(y_true_label_index[1:],
                                               split_idx)
            y_pred_label_index, _ = split_data(y_pred_label_index[1:],
                                               split_idx)
        return balanced_accuracy_score(y_true_label_index, y_pred_label_index)


class Accuracy(rw.score_types.classifier_base.ClassifierBaseScoreType):
    """ Compute accuracy.
    """
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="accuracy", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        is_test_set = (-1 in y_pred_label_index)
        if is_test_set:
            split_idx = y_true_label_index[0]
            y_true_label_index, _ = split_data(y_true_label_index[1:],
                                               split_idx)
            y_pred_label_index, _ = split_data(y_pred_label_index[1:],
                                               split_idx)
        return accuracy_score(y_true_label_index, y_pred_label_index)


class RMSE(rw.score_types.BaseScoreType):
    """ Compute root mean square error.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="rmse", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        is_test_set = np.isnan(y_pred[0])
        if is_test_set:
            split_idx = y_true[0, 0]
            y_pred, _ = split_data(y_pred[1:], split_idx)
            y_true, _ = split_data(y_true[1:], split_idx)
        return np.sqrt(np.mean(np.square(y_true - y_pred)))


class DeepDebiasingMetric(rw.score_types.BaseScoreType):
    """ Compute the final metric as descibed in the paper.
    """
    is_lower_the_better = False
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, score_types, n_sites, name="combined",
                 precision=2):
        self.name = name
        self.score_types = score_types
        self.n_sites = n_sites
        self.precision = precision
        self.score_type_mae_private_age = MAE(
            name="mae_private_age", precision=3)

    def score_function(self, ground_truths_combined, predictions_combined,
                       valid_indexes=None):
        scores = {}
        split_idx = None
        for score_type, ground_truths, predictions in zip(
                self.score_types,
                ground_truths_combined.predictions_list,
                predictions_combined.predictions_list):
            _predictions = predictions.y_pred
            _ground_truths = ground_truths.y_pred
            if _predictions.ndim == 1:
                is_test_set = (
                    (-1 in _predictions) or np.isnan(_predictions[0]))
            else:
                is_test_set = (
                    (-1 in _predictions) or np.isnan(_predictions[0, 0]))
            if is_test_set:
                if split_idx is None:
                    if _ground_truths.ndim == 1:
                        split_idx = _ground_truths[0]
                    else:
                        split_idx = _ground_truths[0, 0]
                internal_predictions, external_predictions = split_data(
                    _predictions[1:], split_idx)
                internal_ground_truths, external_ground_truths = split_data(
                    _ground_truths[1:], split_idx)
                if internal_ground_truths.ndim == 1:
                    scores[self.score_type_mae_private_age.name] = (
                        self.score_type_mae_private_age(
                            external_ground_truths, external_predictions))
                if valid_indexes is None:
                    ground_truths.y_pred = internal_ground_truths
                    predictions.y_pred = internal_predictions
                else:
                    ground_truths.y_pred = np.concatenate(
                        (_ground_truths[:1], internal_ground_truths), axis=0)
                    predictions.y_pred = np.concatenate(
                        (_predictions[:1], internal_predictions), axis=0)
            scores[score_type.name] = score_type.score_function(
                ground_truths, predictions, valid_indexes)
        metric = (
            scores.get("mae_private_age", 0) * scores["bacc_site"] +
            (1. / self.n_sites) * scores["mae_age"])
        return metric

    def __call__(self, y_true, y_pred):
        raise ValueError("Combined score has no deep score function.")


def get_cv(X, y):
    """ Get N folds cross validation indices.
    Remove the index 0 as it corresponds to the header.
    """
    cv_train = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = []
    for cnt, (train_idx, test_idx) in enumerate(cv_train.split(X, y)):
        train_idx = train_idx.tolist()
        if 0 in train_idx:
            train_idx.remove(0)
        test_idx = test_idx.tolist()
        if 0 in test_idx:
            test_idx.remove(0)
        folds.append((np.asarray(train_idx), np.asarray(test_idx)))
        if cnt == 1:
            break
    return folds


def _read_data(path, dataset):
    """ Read data.

    Parameters
    ----------
    path: str
        the data location.
    dataset: str
        'train' or 'test'.

    Returns
    -------
    x_arr: array (n_samples, n_features)
        input data.
    y_arr: array (n_samples, )
        target data.
    """
    df = pd.read_csv(os.path.join(path, "data", dataset + ".tsv"), sep="\t")
    y_arr = df[["age", "site"]].values
    x_arr = np.load(os.path.join(path, "data", dataset + ".npy"),
                    mmap_mode="r")
    return x_arr, y_arr


def get_train_data(path="."):
    """ Get openBHB public train set.
    """
    return _read_data(path, "train")


def get_test_data(path="."):
    """ Get openBHB internal(public)/external(private) test set.

    This external/private test set is unable during local executions of
    the code. This set is used when computing the combined loss. Locally
    this set is replaced by the public test set, and the combined loss may
    not be relevant.
    """
    return _read_data(path, "test")


def _init_from_pred_labels(self, y_pred_labels):
    """ Initalize y_pred to uniform for (positive) labels in y_pred_labels.

    Initialize multiclass Predictions from ground truth. y_pred_labels
    can be a single (positive) label in which case the corresponding
    column gets probability of 1.0. In the case of multilabel (k > 1
    positive labels), the columns corresponing the positive labels
    get probabilities 1/k.

    Parameters
    ----------
    y_pred_labels : list of objects or list of list of objects
        (of the same type)
    """
    type_of_label = type(self.label_names[0])
    self.y_pred = np.zeros(
        (len(y_pred_labels), len(self.label_names)), dtype=np.float64)
    if any(np.isnan(y_pred_labels)):
        self.y_pred[0, 0] = y_pred_labels[0]
        self.y_pred[0, 1:] = np.nan
    for ps_i, label_list in zip(self.y_pred[1:], y_pred_labels[1:]):
        if type(label_list) != np.ndarray and type(label_list) != list:
            label_list = [label_list]
        if not any(np.isnan(label_list)):
            label_list = list(map(type_of_label, label_list))
            for label in label_list:
                ps_i[self.label_names.index(label)] = 1.0 / len(label_list)
        else:
            ps_i[:] = np.nan


@property
def _y_pred_label_index(self):
    """ Multi-class y_pred is the index of the predicted label.
    """
    indices = np.argwhere([any(row) for row in np.isnan(self.y_pred)])
    labels = np.argmax(self.y_pred, axis=1)
    if len(indices) > 0:
        labels[indices] = -1
        if not np.isnan(self.y_pred[0, 0]):
            labels[0] = int(self.y_pred[0, 0])
    return labels


def make_multiclass(label_names=[]):
    Predictions = type(
        'Predictions',
        (rw.prediction_types.multiclass.BasePrediction,),
        {'label_names': label_names,
         'n_columns': len(label_names),
         'n_columns_true': 0,
         '__init__': rw.prediction_types.multiclass._multiclass_init,
         '_init_from_pred_labels': _init_from_pred_labels,
         'y_pred_label_index': _y_pred_label_index,
         'y_pred_label': rw.prediction_types.multiclass._y_pred_label,
         'combine': rw.prediction_types.multiclass._combine,
         })
    return Predictions


problem_title = (
    "Brain age prediction and debiasing with site-effect removal in MRI "
    "through representation learning.")
# _, y = get_train_data()
# _prediction_site_names = sorted(np.unique(y[:, 1]))
# TODO: switch labels manually in prod (configured to wotk with CI in test
# mode)
_prediction_site_names = [0, 1]
# _prediction_site_names = list(range(64))
_target_column_names = ["age", "site"]
Predictions_age = rw.prediction_types.make_regression(
    label_names=[_target_column_names[0]])
Predictions_site = make_multiclass(
    label_names=_prediction_site_names)
Predictions = rw.prediction_types.make_combined(
    [Predictions_age, Predictions_site])
score_type_r2_age = R2(name="r2_age", precision=3)
score_type_mae_age = MAE(name="mae_age", precision=3)
score_type_rmse_age = RMSE(name="rmse_age", precision=3)
score_type_acc_site = Accuracy(name="acc_site", precision=3)
score_type_bacc_site = BACC(name="bacc_site", precision=3)
score_type_ext_mae_age = ExtMAE(name="ext_mae_age", precision=3)

score_types = [
    DeepDebiasingMetric(
        name="challenge_metric", precision=3,
        n_sites=len(_prediction_site_names),
        score_types=[score_type_mae_age, score_type_bacc_site]),
    rw.score_types.MakeCombined(score_type=score_type_r2_age, index=0),
    rw.score_types.MakeCombined(score_type=score_type_mae_age, index=0),
    rw.score_types.MakeCombined(score_type=score_type_rmse_age, index=0),
    rw.score_types.MakeCombined(score_type=score_type_acc_site, index=1),
    rw.score_types.MakeCombined(score_type=score_type_bacc_site, index=1),
    rw.score_types.MakeCombined(score_type=score_type_ext_mae_age, index=0)
]
workflow = DeepDebiasingEstimator(
    filename="estimator.py",
    additional_filenames=["weights.pth", "metadata.pkl"])
