import pandas as pd
import numpy as np
from scipy import stats
import numpy.ma as ma


class MissingValuesPopulator():
    def __init__(self):
        self.mColumnMeans = []
    def transform(self, X):
        print("Filling in missing values...")
        Xcopy = X.copy()
        for col, value in self.mColumnMeans:
            Xcopy[col].fillna(value, inplace=True)
        return Xcopy 
    def fit(self, X, y=None):
        for col in list(X):
            if X[col].dtype == np.object:
                self.mColumnMeans.append( (col, "N/A") )
            else:
                self.mColumnMeans.append( (col, X[col].mean()) )
        return self

class CorrelationIncreaser():
    def __init__(self):
        self.mTransfomations = []
    def transform(self, X):
        print("Applying correlation increasing functions...")
        for index, func in self.mTransfomations:
            X[:, index] = func(X[:, index])
            X[:, index][abs(X[:, index]) == np.inf] = np.nan
            X[:, index] = np.where( ~np.isfinite(X[:, index]), np.nanmean(X[:, index]), X[:, index])  
        return X
    def fit(self, X, y=None):
        print("Discovering correlation increasing functions...")
        functions = [np.log, np.square, np.sqrt, self.cube, self.cubeRoot]
        for i in range(X.shape[1]):
            bestCorr = abs( stats.pearsonr( X[:, i], y)[0] )
            bestFunc = None
            for func in functions:
                mapped = func(X[:, i])
                if np.isfinite(mapped).all():
                    corr = abs( stats.pearsonr( mapped, y )[0] )
                    if np.isfinite(corr) and bestCorr < corr:
                        bestCorr = corr
                        bestFunc = func
            if bestFunc is not None:
                self.mTransfomations.append( (i, bestFunc) )
        return self
    def cube(self, x):
        return np.power(x, 3)
    def cubeRoot(self, x):
        return np.power(x, 1/3)

class Information():
    def transform(self, X):
        print("TRANSFORM")
        return X
    def fit(self, X, y=None):
        print("FIT")
        return self