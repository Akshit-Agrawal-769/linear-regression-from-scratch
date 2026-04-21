import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class bayesianLR:
    def __init__(self,alpha,sigma):
        self.alpha=alpha
        self.beta=1/sigma**2
        # prior_mean=0
        # prior_uncertainity=(1/alpha)*I

    def fit(self,X,y):
        I = np.eye(X.shape[1])
        I[0][0]=0
        self.Lambda = self.alpha * I  + (self.beta) * X.T @ X
        self.m_N = (self.beta) * np.linalg.solve(self.Lambda, X.T @ y)

    def predict(self, X, return_std=False):
        # mean prediction — same as always
        y_mean = X @ self.m_N   #(n,featues  @  features,1)
    
        # predictive variance for each point
        # = epistemic (weight uncertainty) + aleatoric (noise)
        self.S_n = np.linalg.inv(self.Lambda)
        y_var = 1/self.beta +np.sum((X @ self.S_n) * X , axis=1)
     
        return y_mean, np.sqrt(y_var)