
from features import FeatureTrajectory
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from sklearn import *
import sklearn
import numpy as np
import copy, re
import inspect
import IPython

def _flatten(X, y=None):
    n_seq = len(X)

    if isinstance(X, FeatureTrajectory):
        X = [X]

    if isinstance(X, list) and isinstance(X[0], FeatureTrajectory):
        X = [seq.features for seq in X]

    # this should handle different length trajectories as well
    # (n_seq, n_frames, n_waters, n_features)
    Xflat = [np.concatenate(frame) for frame in X]
    # (n_seq, n_frames * n_waters, n_features)
    Xflat = np.concatenate(Xflat)
    # (n_seq * n_frames * n_waters, n_features)

    if not y is None:
        yflat = np.concatenate(y)
        return Xflat, yflat

    return Xflat, None


def _expand(Xflat, n_seq, n_frames, n_particles, yflat=None):

    n_features = Xflat.shape[1]

    if isinstance(n_frames, int):
        n_frames = [n_frames] * n_seq

    X = []
    y = []
    for i in xrange(n_seq):

        if i == 0:
            a = 0
        else:
            a = np.sum(n_frames[:i] * n_particles)

        b = a + n_frames[i] * n_particles

        seq = Xflat[a:b].reshape((n_frames[i], n_particles, n_features))
        X.append(seq)

        if not yflat is None:
            y.append(yflat[a:b].reshape((n_frames[i], n_particles)))

    if not yflat is None:
        return X, y

    return X, None


def fit(self, X, y=None):

    Xflat, yflat = _flatten(X, y=y)

    return super(self.__class__, self).fit(Xflat, y=yflat)


def transform(self, X):

    feature_traj = False
    n_seq = len(X)
    n_frames = [len(f) for f in X]
    if isinstance(X[0], FeatureTrajectory):
        features_traj = True
        n_particles = X[0].n_molecules
    else:  # assume it's a np.ndarray
        n_particles = X[0].shape[1] 
    
    Xflat, yflat = _flatten(X, y=None)

    transXflat = super(self.__class__, self).transform(Xflat)

    transX, y = _expand(transXflat, n_seq, n_frames, n_particles, yflat=None)

    if feature_traj:
        transX = [FeatureTrajectory(transX[i], X[i].neighbors) for i in xrange(n_seq)]
    
    return transX


def predict(self, X, y=None):
    
    n_seq = len(X)
    n_frames = [len(f) for f in X]
    if isinstance(X[0], FeatureTrajectory):
        n_particles = X[0].n_molecules
    else:  # assume it's a np.ndarray
        n_particles = X[0].shape[1] 

    Xflat, yflat = _flatten(X, y=y)

    ypredflat = super(self.__class__, self).predict(Xflat, y=yflat)

    X, ypred = _expand(Xflat, n_seq, n_frames, n_particles, yflat=ypredflat)

    return ypred


def score(self, X, y=None):
    Xflat, yflat = _flatten(X, y=y)

    return super(self.__class__, self).score(Xflat, y=yflat)


def get_estimators(obj):
    res = []
    if inspect.ismodule(obj):
        if re.search('^sklearn', obj.__name__):
            for name1, obj1 in inspect.getmembers(obj):
                res.extend(get_estimators(obj1))

    elif inspect.isclass(obj):
        if issubclass(obj, BaseEstimator):
            res.extend([obj])

    return res

# This is every sklearn class that is a base estimator
classes = list(set(get_estimators(sklearn)))

for cls in classes:
    if cls.__name__[0] == '_':
        name = '_Particle%s' % cls.__name__[1:]
    else:
        name = 'Particle%s' % cls.__name__

    cls_dict = {}
    cls_dict['__doc__'] = cls.__doc__
    cls_dict['fit'] = fit
    if issubclass(cls, TransformerMixin):
        cls_dict['transform'] = transform

    if issubclass(cls, RegressorMixin):
        cls_dict['predict'] = predict
        cls_dict['score'] = score

    vars()[name] = type(name, (cls,), cls_dict)

crnt_vars = copy.copy(vars())
__all__ = [cls for cls in crnt_vars if re.search('^Particle', cls)]

