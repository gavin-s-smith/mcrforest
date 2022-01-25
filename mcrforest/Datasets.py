
import pandas as pd
import numpy as np

# A very small demo dataset based on the Synthetic Data from:
# # Smith, G., Mansilla, R. and Goulding, J. "Model Class Reliance for Random Forests". 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.
def get_demo_dataset():
    X = pd.DataFrame([[1, 0, 0],[0, 0, 0],[1, 0, 0],[0, 0, 0],[0, 1, 1],[0, 0, 0],[0, 1, 1],[1, 1, 1],[0, 1, 1],[1, 1, 1],[1, 0, 0],[1, 1, 1],[0, 1, 1],[1, 0, 0],[0, 0, 0],[1, 0, 0],[0, 1, 1],[0, 1, 1],[0, 0, 0],[1, 0, 0],[1, 1, 1],[0, 0, 0],[0, 0, 0],[1, 1, 1],[0, 1, 1],[0, 1, 1],[1, 0, 0],[1, 0, 0],[1, 1, 1],[0, 1, 1],[1, 1, 1],[0, 1, 1],[1, 0, 0],[0, 1, 1],[1, 1, 1],[0, 1, 1],[1, 0, 0],[0, 0, 0],[0, 0, 0],[1, 1, 1],[0, 1, 1],[1, 0, 0],[0, 0, 0],[0, 0, 0],[0, 1, 1],[0, 1, 1],[1, 0, 0],[1, 0, 0],[1, 1, 1],[0, 0, 0]], columns=['A','B','C'])
    y = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    return X, y