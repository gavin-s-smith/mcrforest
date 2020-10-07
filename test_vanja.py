import numpy as np
import pandas as pd
from mcrforest.forest import RandomForestClassifier


X_train = pd.read_csv('/home/gavin/Vanja/X_train.csv')
X_test = pd.read_csv('/home/gavin/Vanja/X_test.csv')
y_train = pd.read_csv('/home/gavin/Vanja/y_train.csv')
y_test = pd.read_csv('/home/gavin/Vanja/y_test.csv')

params = {'bootstrap': False,
 'ccp_alpha': 0.0,
 'class_weight': 'balanced',
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'sqrt',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 2,#10
 'min_samples_split': 11,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 500,
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False,
 'random_state':13111985,
 'mcr_tree_equivilient_tol': 0.0001, 'performance_equivilence': True}

# Define the result array
results = []
#n_estimators=100, max_features=1, max_depth=5, min_samples_split=2, bootstrap=False
rfd = RandomForestClassifier(**params)

y_train[y_train == 1] = 0
y_train[y_train == 2] = 1

rfd.fit(X_train.values,y_train.values.flatten())

print('Score: {}'.format(rfd.score(X_train.values, y_train.values.flatten())))

groups_of_indicies_to_permute = [ [0,1],[2,3,4,5,6,7], [8,9,10], [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21,22,23],[24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],[42,43],[44,45],[46,47],
 [48,49],[50,51,52], [53,54], [56,57], [58,59,60]]


#groups_of_indicies_to_permute = [[1],[4]]

# New MCR+ perm imp
for gp in groups_of_indicies_to_permute:
    rn = rfd.mcr(X_train.values,y_train.values.flatten(), np.asarray(gp) ,  num_times = 10, mcr_type = 1, mcr_as_ratio = False)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR+', rn])


# New MCR- perm imp
for gp in groups_of_indicies_to_permute:
    rn = rfd.mcr(X_train.values,y_train.values.flatten(), np.asarray(gp) ,  num_times = 10,  mcr_type = -1, mcr_as_ratio = False)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR-', rn])

lbl = [ x[0] for x in results if 'MCR+' in x[1] ]
mcrp = [ x[2] for x in results if 'MCR+' in x[1] ]
mcrm = [ x[2] for x in results if 'MCR-' in x[1] ]

print('MCR+ sum: {}'.format(sum(mcrp)))

rf_results2 = pd.DataFrame({'variable':lbl, 'MCR+':mcrp, 'MCR-':mcrm})

import seaborn as sns
import matplotlib.pyplot as plt

def plot_mcr(df_in, fig_size = (11.7, 8.27)):
    df_in = df_in.copy()
    df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns]
    ax = sns.barplot(x='MCR- (lollypops) | MCR+ (bars)',y='variable',data=df_in)
    plt.gcf().set_size_inches(fig_size)
    plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='skyblue')
    plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'skyblue')

plot_mcr(rf_results2)

plt.show()