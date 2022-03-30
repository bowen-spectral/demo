

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris # Load Iris Data

data = pd.read_csv("./dataset/holdout-data-for-predictions.csv")

data = data.drop(columns=["borrower", "loan_block_time", "loan_underlying_symbol"])

y = data["target"]
X = data.drop(columns=["target"])

# Splitting the Dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.13, random_state= 101)

X = X_test.values[:]
Yt_expected = y_test.values[:].reshape(-1, 1)

np.save('dataset/X_first.npy',X)
np.save('dataset/Y_first.npy',Yt_expected)
