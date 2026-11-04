# pls_numpy.py  -  loads the face library's emotion model without scikit-learn.
# The model (models/model_emotion_pls=30_fullfeatures=False.joblib) is a scikit-learn 0.23 PLSRegression. Windows
# Smart App Control blocks scikit-learn's compiled files on this PC, and newer scikit-learn versions changed the
# PLS internals anyway. A fitted PLS model is only a linear map, so the file is unpickled with a stand-in class and
# predicted with numpy, exactly as scikit-learn 0.23 did:  ((X - x_mean_) / x_std_) @ coef_ + y_mean_
import pickle

import numpy as np
from joblib.numpy_pickle import NumpyUnpickler


class StandIn:
    # receives the saved attributes of any scikit-learn object in the file
    def __setstate__(self, state):
        self.__dict__.update(state)


class PLSNumpy:
    def __init__(self, pls):
        self.x_mean = np.asarray(pls.x_mean_, dtype=np.float64)
        self.x_std = np.asarray(pls.x_std_, dtype=np.float64)
        self.y_mean = np.asarray(pls.y_mean_, dtype=np.float64)
        self.coef = np.asarray(pls.coef_, dtype=np.float64)                  # (n_features, n_targets) in 0.23

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return ((X - self.x_mean) / self.x_std) @ self.coef + self.y_mean


class _Unpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module.startswith("sklearn"):
            return StandIn
        return super().find_class(module, name)


def load(path):
    # drop-in for joblib.load on the emotion model file
    with open(path, "rb") as f:
        out = _Unpickler(path, f, ensure_native_byte_order=False).load()   # plain (uncompressed) joblib file
    if isinstance(out, dict) and "model" in out:
        out["model"] = PLSNumpy(out["model"])
    return out


if __name__ == "__main__":
    m = load("facial_expression_analysis/models/model_emotion_pls=30_fullfeatures=False.joblib")
    p = m["model"]
    print({k: v for k, v in m.items() if k != "model"}, "coef", p.coef.shape, "x_mean", p.x_mean.shape)
