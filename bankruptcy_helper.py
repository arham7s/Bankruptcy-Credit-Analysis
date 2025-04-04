import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from sklearn.decomposition import PCA

import matplotlib as mpl
class Helper():
    def __init__(self):
        return
    
    def bankruptcy_PCA(self, X, n_components=0.95, **params):
        """
        Fit PCA to X

        Parameters
        ----------
        n_components: number of components
        - Passed through to sklearn PCA
        -- <1: interpreted as fraction of explained variance desired
        -- >=1 interpreted as number of components
        
        """
        if n_components is not None:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA()

        pca.fit(X)

        return pca

    def transform(self, X,  model):
        """
        Transform samples through sklearn model

        Parameters
        ----------
        X: ndarray (num_samples, num_features)
        model: sklearn model object, e.g, PCA

        X_reduced: ndarray (num_samples, pca.num_components_)
        """
        X_transformed = model.transform(X)
        return X_transformed

    def inverse_transform(self, X,  model):
        """
        Invert  samples that were transformed through sklearn model

        Parameters
        ----------
        X: ndarray (num_samples, num_features_trasnformed)
        model: sklearn model object, e.g, PCA

        X_reconstruct: ndarray (num_samples, num_features)
        """
        X_reconstruct = model.inverse_transform(X)
        return X_reconstruct

    def num_components_for_cum_variance(self, pca, thresh):
        """
        Return number of components of PCA such that cumulative variance explained exceeds threshhold

        Parameters
        ----------
        pca: PCA object
        thresh: float. Fraction of explained variance threshold
        """

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= thresh) + 1

        return d

    def plot_cum_variance(self, pca):
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        x  = range(1, 1 + cumsum.shape[0])
        
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        _ = ax.plot(x, cumsum)

        _ = ax.set_title("Cumulative variance explained")
        _ = ax.set_xlabel("# of components")
        _ = ax.set_ylabel("Fraction total variance")

        _= ax.set_yticks( np.linspace(0,1,11)  )

        return fig, ax


    
