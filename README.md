# mcrforest

This repository contains the code from our 2020 NeurIPS paper:

*Smith, G., Mansilla, R. and Goulding, J. "Model Class Reliance for Random Forests". 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.*

[Paper](https://proceedings.neurips.cc/paper/2020/hash/fd512441a1a791770a6fa573d688bff5-Abstract.html) | [Supplementary Material](https://proceedings.neurips.cc/paper/2020/file/fd512441a1a791770a6fa573d688bff5-Supplemental.pdf) | [3 Minute Explainer Video](https://slideslive.com/38937760/model-class-reliance-mcr-for-random-forests)

## Installation
Install for use via pip:
```
pip install git+https://github.com/gavin-s-smith/mcrforest
```
Reinstall via pip:
```
pip install --upgrade --no-cache-dir git+https://github.com/gavin-s-smith/mcrforest
```

Install for developing / debugging from unzipped source:
```
python setup.py build_ext --inplace
```

## Usage

mcrforest is an extention to the sklearn RandomForestClassifier and RandomForestRegressor classes and can be used as a direct replacement.

mcrforest includes two additional methods which can be called after training a model. In addition is one restriction on the model building that must be met, that bootstrap must be set to false.

```
mcr(X_in, y_in, indices_to_permute, num_times = 100, mcr_type = 1, seed = 13111985)
```

Computes and MCR+ or MCR- score of a variable or group of variables.

*Parameters:*
```
X_in (2D numpy array): The input features to use to compute the MCR.
y_in (1D numpy array): The output features to use to compute the MCR.
indices_to_permute (1D numpy array): A numpy array of the index/indices indicating the variable or group of variables to compute MCR for.
num_times (int): The number of times to permute the index/indices.
mcr_type (int): 1 for MCR+, -1 for MCR-.
seed (int): A seed to control the permutation randomness.
```

```
plot_mcr(X_in, y_in, feature_names = None, feature_groups_of_interest = 'all individual features', num_times = 100, show_fig = True)
```

Compute the required information for an MCR plot and optionally display the MCR plot.

*Parameters:*
```
X_in : {numpy array or Pandas DataFrame} of shape (n_samples, n_features)
    The input samples. 
y_in : {numpy array or Pandas DataFrame} of shape (n_samples)
    The output values.
feature_names : {array-like} of shape (n_features)
    A list or array of the feature names. If None and a DataFrame is passed the feature names will be taken from the DataFrame else features will be named using numbers.
feature_groups_of_interest : {str or numpy array of numpy arrays}
    Either:
    1. 'all individual features': compute the MCR+/- for all features individually. Equvilent to: [[x] for x in range(len(feature_names))]
    2. A numpy array where each element is a numpy array of variable indexes which will be jointly permuated (i.e. these indexes will be considered a single unit of analysis for MCR)
       A single MCR+ and single MCR- score (plotted as a single bar in the graph) will be computed for each sub-array.
num_times : int
        The number of permutations to use when computing the MCR.
show_fig : bool
        If True show the MCR graph. In either case a dataframe with the information that would have been shown in the graph is returned.

Returns
-------
rf_results2 : {pandas DataFrame} of shape (2*[number_of_features OR len(feature_groups_of_interest)], 3)
        A DataFrame with three columns: ['variable', 'MCR+', 'MCR-']
        Where the column variable contains the variable name and the columns MCR+ and MCR- contain the variable's MCR+ and MCR- scores respectively.
```

### Example Usage A
*NOTE if using a large dataset and experiencing slow runtimes:* 
While the computational complexity is no different, currently in practice the building of MCR Random Forests can be slower than the sklearn version. 
As such, when determining the best meta-paramters (controling the complexity) you may want to use the sklearn version and then re-train an MCR Forest 
version as a final step before calling the MCR methods. This is valid as the MCR Random Forest is a direct extenstion of the sklearn version.
This is demonstrated below. If speed is not important or you have a small dataset you can use the RandomForestRegressor or RandomForestClassifier
from the mcrforest package instead of the sklearn one below.
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestRegressor as mcrRandomForestRegressor
from sklearn.ensemble import RandomForestRegressor as sklearnRandomForestRegressor

# Load data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# If we are going to use the training set for computing MCR then we MUST ensure 
# we have controlled the complexity of the fit. This is equally true for traditional
# permutation importance.
# See: https://christophm.github.io/interpretable-ml-book/feature-importance.html

# Determine the best meta-parameters using the sklearn Random Forest (see note above, could equally use the RandomForest from the mcrforest package)
base_model = sklearnRandomForestRegressor(random_state = 13111985, bootstrap=False)

search = {'n_estimators':[500],'max_features':['auto'], 'max_depth':[5,10,15,20,30]}

rf_cv_model = GridSearchCV(base_model, search)
rf_cv_model.fit(X_train,y_train)

# Refit a RandomForest from the mcrforest package with best parameters (see note above)
model = mcrRandomForestRegressor( **rf_cv_model.best_params_ )

model.plot_mcr(X_train, y_train)

```




### Example Usage B
Example Usage B shows how to compute the MCR+/- values and then plot them without the help of `plot_mcr(.)`.
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestRegressor

# Load data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# If we are going to use the training set for computing MCR then we MUST ensure 
# we have controlled the complexity of the fit. This is equally true for traditional
# permutation importance.
# See: https://christophm.github.io/interpretable-ml-book/feature-importance.html

base_model = RandomForestRegressor(random_state = 13111985, bootstrap=False)

search = {'n_estimators':[500],'max_features':['auto'], 'max_depth':[5,10,15,20,30]}

rf_cv_model = GridSearchCV(base_model, search)
rf_cv_model.fit(X_train,y_train)

# Refit with best parameters
model = RandomForestRegressor( **rf_cv_model.best_params_ )

# Compute MCR+ for each variable
results = []
groups_of_indicies_to_permute = [[x] for x in range(len(X_train.columns))]

for gp in groups_of_indicies_to_permute:
    rn = model.mcr(X_train.values,y_train.values, np.asarray(gp) ,  num_times = 20, mcr_type = 1)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR+', rn])


# Compute MCR- for each variable
for gp in groups_of_indicies_to_permute:
    rn = model.mcr(X_train.values,y_train.values, np.asarray(gp) ,  num_times = 20,  mcr_type = -1)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR-', rn])


# Plot the results

lbl = [ x[0] for x in results if 'MCR+' in x[1] ]
mcrp = [ x[2] for x in results if 'MCR+' in x[1] ]
mcrm = [ x[2] for x in results if 'MCR-' in x[1] ]

rf_results = pd.DataFrame({'variable':lbl, 'MCR+':mcrp, 'MCR-':mcrm})


def plot_mcr(df_in, fig_size = (11.7, 8.27)):
    df_in = df_in.copy()
    df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns]
    ax = sns.barplot(x='MCR- (lollypops) | MCR+ (bars)',y='variable',data=df_in)
    plt.gcf().set_size_inches(fig_size)
    plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='skyblue')
    plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'skyblue')

plot_mcr(rf_results)
plt.show()
```

### Example output plot if using the code above
![Example image](http://cs.nott.ac.uk/~pszgss/research/mcr/examplemcr.png)


### Replication of results from the paper
Synthetic Experiments:
https://colab.research.google.com/drive/1UuORvqSYW14eiBX3nzz2WWUrjQAXcvFw

COMPAS Experiments:
https://colab.research.google.com/drive/1-hWJ4DNOnvrLz4fxGd--NJjGHV26TIei

Breast Cancer Experiments
https://colab.research.google.com/drive/16HGlytaraR6Kn4EmqKk0_Q9Nl7O_hU5F

RF-MCR Analysis:
https://colab.research.google.com/drive/1AMDW9Ss69QEzgBkMgx8Tw_zIpcZnMcr4



## Common errors
`AttributeError: 'RandomForestRegressor' object has no attribute '_validate_data'`
mcrforest requires sklearn version >= 0.23. Ensure you reinstall mcrforest after upgrading.

### Windows Only
`Cannot open include file: 'basetsd.h': No such file or directory`
You need to ensure you have the correct compilers for Windows installed since the package uses Cython.
Install the correct compilers for your version of python from: https://wiki.python.org/moin/WindowsCompilers