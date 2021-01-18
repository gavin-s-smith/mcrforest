#pip install --upgrade --no-cache-dir git+https://github.com/gavin-s-smith/mcrforest
#============= Import libraries ==============#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
#from IPython.display import clear_output
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import seaborn as sns
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestRegressor
#from IPython.display import clear_output
import tkinter as tk

#============= Import Dataset ==============#
# This is compas_X.csv in the supplementary zip file
X = pd.read_csv('./compas_X.csv') 
# This is compas_y.csv in the supplementary zip file
y = pd.read_csv('./compas_y.csv',header=0,names=['Y']) 
# This is compas_train_indices.csv in the supplementary zip file
train_bool_mask = pd.read_csv('./compas_train_indices.csv').values.flatten()

X_train = X[train_bool_mask]
y_train = y[train_bool_mask]
X_test = X[~train_bool_mask]
y_test = y[~train_bool_mask]

X = X_train
y = y_train

debug_call = False

# For the model class RF
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# For the model class RF
# 3.1713
params = {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': 1,
 'max_leaf_nodes': None, 'max_samples': None, 'mcr_tree_equivilient_tol': 3.1713, 'min_impurity_decrease': 0.01, 
 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 84, 'n_jobs': None, 'oob_score': False, 
 'performance_equivilence': True, 'random_state': 13111985, 'spoof_as_sklearn': False, 'verbose': 0, 'warm_start': False}
model = RandomForestRegressor(**params)
model.fit(X, y.values.flatten())

# # For the model class RF
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # For the model class RF
# search = {'n_estimators':[100],'min_impurity_decrease':[0.001,0.01],'max_features':[1,'auto']}
# rf_cv_model = GridSearchCV(RandomForestRegressor(mcr_tree_equivilient_tol=3.1713, bootstrap=False, 
#                                             criterion='mse', random_state=13111985), search, cv = kf, refit = True)
# model = rf_cv_model.fit(X, y.values.flatten()).best_estimator_

#============= Create Datframe with MCR- & MCR+ ==============#

must_use_variable_ordering = np.asarray([4,0,1,2,3])
mcr_p4 = model.mcr( X.values,y.values.flatten(), np.asarray([4]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = debug_call, mcr_type = 1, mcr_as_ratio=False)
must_use_variable_ordering = np.asarray([0,1,2,3,4])
mcr_m4 = model.mcr( X.values,y.values.flatten(), np.asarray([4]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = debug_call, mcr_type = -1, mcr_as_ratio=False)


# must_use_variable_ordering = np.asarray([2,0,1,3,4])
# mcr_p2 = model.mcr( X.values,y.values.flatten(), np.asarray([2]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False,  mcr_type = 1, mcr_as_ratio=False)
# must_use_variable_ordering = np.asarray([0,1,3,4,2])
# mcr_m2 = model.mcr( X.values,y.values.flatten(), np.asarray([2]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = -1, mcr_as_ratio=False)

# must_use_variable_ordering = np.asarray([0,1,2,3,4])
# mcr_p0 = model.mcr( X.values,y.values.flatten(), np.asarray([0]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = 1, mcr_as_ratio=False)
# mcr_p0_o = model.mcr( X.values,y.values.flatten(), np.asarray([0]), mcr_ordering = None, num_times = 20, debug = False, mcr_type = 1, mcr_as_ratio=False)
# must_use_variable_ordering = np.asarray([1,2,3,4,0])
# mcr_m0 = model.mcr( X.values,y.values.flatten(), np.asarray([0]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = -1, mcr_as_ratio=False)
# mcr_m0_o = model.mcr( X.values,y.values.flatten(), np.asarray([0]), mcr_ordering = None, num_times = 20, debug = False, mcr_type = -1, mcr_as_ratio=False)


print(f'MCR+/-[4]: {mcr_m4} <-> {mcr_p4} ')
# print(f'MCR+/-[2]: {mcr_m2} <-> {mcr_p2} ')
# print(f'MCR+/-[0]: {mcr_m0} <-> {mcr_p0} ')
#exit(-1)
print('USE(4)')
model.set_estimators(True, 4 )

must_use_variable_ordering = np.asarray([4,0,1,2,3])
r_p4 = model.mcr(X.values,y.values.flatten(), np.asarray([4]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = 1, 
                    mcr_as_ratio=False, enable_Tplus_transform = True)
must_use_variable_ordering = np.asarray([4,0,1,2,3])
r_m4 = model.mcr(X.values,y.values.flatten(), np.asarray([4]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = -1, 
                    mcr_as_ratio=False, enable_Tplus_transform = True)

# must_use_variable_ordering = np.asarray([4,2,0,1,3])
# r_p2 = model.mcr(X.values,y.values.flatten(), np.asarray([2]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = 1, mcr_as_ratio=False)
# must_use_variable_ordering = np.asarray([4,0,1,3,2])
# r_m2 = model.mcr(X.values,y.values.flatten(), np.asarray([2]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = -1, mcr_as_ratio=False)

# must_use_variable_ordering = np.asarray([4,0,1,2,3])
# r_p0 = model.mcr(X.values,y.values.flatten(), np.asarray([0]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = 1, mcr_as_ratio=False)
# must_use_variable_ordering = np.asarray([4,1,2,3,0])
# r_m0 = model.mcr(X.values,y.values.flatten(), np.asarray([0]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug = False, mcr_type = -1, mcr_as_ratio=False)

print(f'MCR+/-[4]: {r_m4} <-> {r_p4} ')
# print(f'MCR+/-[2]: {r_m2} <-> {r_p2} ')
# print(f'MCR+/-[0]: {r_m0} <-> {r_p0} ')


# print('AVOID(2)')