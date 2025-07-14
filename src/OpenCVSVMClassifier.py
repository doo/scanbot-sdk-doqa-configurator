import tempfile
from pathlib import Path

import numpy as np
import cv2
from sklearn.base import BaseEstimator, ClassifierMixin


class OpenCVSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.svm_ = None  # Read by scikit-learn's check_is_fitted

    def _get_kernel_type(self):
        kernel_map = {
            'linear': cv2.ml.SVM_LINEAR,
            'poly': cv2.ml.SVM_POLY,
            'rbf': cv2.ml.SVM_RBF,
            'sigmoid': cv2.ml.SVM_SIGMOID
        }
        return kernel_map.get(self.kernel, cv2.ml.SVM_LINEAR)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32).reshape(-1, 1)

        self.svm_ = cv2.ml.SVM_create()
        self.svm_.setType(cv2.ml.SVM_C_SVC)
        self.svm_.setKernel(self._get_kernel_type())
        self.svm_.setC(1.0)
        self.svm_.setDegree(3)
        self.svm_.setGamma(1 / (X.shape[0] * X.var()))

        self.svm_.train(X, cv2.ml.ROW_SAMPLE, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        _, y_pred = self.svm_.predict(X)
        return y_pred.ravel()

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            "kernel": self.kernel,
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
