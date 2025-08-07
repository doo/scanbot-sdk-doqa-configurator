import tempfile
from pathlib import Path

import cv2
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class OpenCVSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel: str = "", C: float = 0.0, gamma_factor: float = 0.0):
        self.kernel = kernel
        self.C = C
        self.gamma_factor = gamma_factor
        self.svm_ = None  # Read by scikit-learn's check_is_fitted

    def _get_kernel_type(self):
        kernel_map = {
            'linear': cv2.ml.SVM_LINEAR,
            'poly': cv2.ml.SVM_POLY,
            'rbf': cv2.ml.SVM_RBF,
            'sigmoid': cv2.ml.SVM_SIGMOID,
        }
        return kernel_map.get(self.kernel, cv2.ml.SVM_LINEAR)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32).reshape(-1, 1)

        self.svm_ = cv2.ml.SVM_create()
        self.svm_.setType(cv2.ml.SVM_C_SVC)
        self.svm_.setKernel(self._get_kernel_type())
        assert self.C is not None, "C must be set before fitting"
        self.svm_.setC(self.C)
        self.svm_.setDegree(3)
        assert self.gamma_factor is not None, "Gamma must be set before fitting"
        gamma = self.gamma_factor / (X.shape[0] * X.var())
        self.svm_.setGamma(gamma)

        num_positive = np.sum(y == 1)
        num_negative = np.sum(y == 0)
        if num_positive == 0 or num_negative == 0:
            raise ValueError("Training data does not contain enough good or bad samples")
        class_weights = np.array([1.0, num_negative / num_positive], dtype=np.float32)
        self.svm_.setClassWeights(class_weights)

        self.svm_.train(X, cv2.ml.ROW_SAMPLE, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        _, y_pred = self.svm_.predict(X)
        return y_pred.ravel()

    def get_num_support_vectors(self):
        assert self.svm_ is not None, "Model must be fitted before getting support vectors"
        return self.svm_.getSupportVectors().shape[0]

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma_factor": self.gamma_factor,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def export(self):
        assert self.svm_ is not None, "Model must be fitted before export"
        tmp_filename = tempfile.mktemp(suffix='.xml')
        self.svm_.save(tmp_filename)
        xml = Path(tmp_filename).read_text()
        Path(tmp_filename).unlink()
        return dict(SVM=xml)
