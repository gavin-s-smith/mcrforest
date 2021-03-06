"""
Forest of trees-based ensemble methods.

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause


import numbers
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel, delayed
from sklearn.base import is_classifier
from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from .tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                    ExtraTreeClassifier, ExtraTreeRegressor)
from ._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from ._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import permutations
import pandas as pd

__all__ = ["RandomForestClassifier",
           "RandomForestRegressor"]
        #    ,
        #    "ExtraTreesClassifier",
        #    "ExtraTreesRegressor",
        #    "RandomTreesEmbedding"]

MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples,
                                              n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None,
                          n_samples_bootstrap=None):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples,
                                           n_samples_bootstrap)
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y,
                                                            indices=indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y,
                                                        indices=indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree

# GAVIN TODO: use sklearn permutation
def inplace_permute_or_random_sample_with_replacement( X, column_idx, permute = 'permute' ):
    
    
    if permute == 'permute':
        #print('Using permute')
        np.random.shuffle(X[:, column_idx]) #shuffle
    elif permute == 'uniform':
        #print('Using uniform')
        X[:, column_idx] = np.random.choice( np.unique(X[:, column_idx]), size = X.shape[0], replace = True)
    elif permute == 'conditional':
        #print('WARNING: Using an exact condition permutation scheme. This should only be used when you have discrete inputs.')
        inplace_permute_or_random_sample_with_replacement_conditional(X,column_idx)
    else:
        raise Exception('Not implemented alsdkjflasdfjasldfj')

def inplace_permute_or_random_sample_with_replacement_conditional( X, column_idx ):
    
    other_cols = np.delete(X, column_idx, axis=1)
    #new_array = [tuple(row) for row in other_cols]
    uniques = np.unique(other_cols,axis=0)
    if len(uniques.shape) == 1:
        # degenerate case, reshape
        tmp = np.zeros((1,len(uniques)))
        tmp[0,:] = uniques
        uniques = tmp

    for row in uniques:
        #shaped_row = np.zeros((1,other_cols.shape[1]))
        #shaped_row[0,:] = row
        mask = np.all(other_cols == row,axis=1)
        X[mask,column_idx] = np.random.permutation(X[mask,column_idx]) 

class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 max_samples=None,
                 mcr_tree_equivilient_tol = 0.00001, performance_equivilence = True):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.mcr_tree_equivilient_tol = mcr_tree_equivilient_tol
        self.performance_equivilence = performance_equivilence

    def print_trees(self, col_names):

        for i,e in enumerate(self.estimators_):
            print('\nTree Number from forest: {}\n'.format(i))
            if is_classifier(e):
                print('Probabily table to class mapping: {}'.format( e.classes_) )
            e.tree_.print_tree(col_names)


    def prob2predictions(self, proba):

        n_samples = proba.shape[0]

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            
            class_type = self.classes_[0].dtype
            predictions = np.zeros((n_samples, self.n_outputs_),
                                    dtype=class_type)
            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[:, k], axis=1),
                    axis=0)

            return predictions

    def predict_tree(self, tree_estimator, X ):
        rtn = tree_estimator.predict(X)
            
        if is_classifier(self):
            # By default a sklearn RF will first convert the input classes to 0,1,2,3... etc.
            # These will then be what the trees are trained off. I.e. y is mapped at the forest level (inputs 1,2) and trees learn and predict (0,1)
            # Since here we are at the forest level we must map it back
            rtn = self.classes_.take(rtn.astype(np.int64),axis=0)    

        return rtn
    
    def predict_vim_tree(self, tree_estimator, X_perm, indices_to_permute, mcr_type):
        rtn = tree_estimator.predict_vim(X_perm, indices_to_permute, mcr_type)
            
        if is_classifier(self):
            # By default a sklearn RF will first convert the input classes to 0,1,2,3... etc.
            # These will then be what the trees are trained off. I.e. y is mapped at the forest level (inputs 1,2) and trees learn and predict (0,1)
            # Since here we are at the forest level we must map it back
            rtn = self.classes_.take(rtn.astype(np.int64),axis=0)    

        return rtn

# GAVIN TODO: Integrate better
    def mcr(self, X_in, y_in, indices_to_permute, e_switch = False, 
                                    num_times = 100, debug = False, 
                                    mcr_type = 1, restrict_trees_to = None, mcr_as_ratio = False, seed = 13111985, 
                                    enable_Tplus_transform = True
                                    ):
        
        is_classification = is_classifier(self)

        # if (windows) passes an int32 who cares, we'll just upgrade it to int64 for the cython code. If we don't have integers though, throw an exception
        if not indices_to_permute.dtype.kind in np.typecodes["AllInteger"]:
                raise Exception('indices_to_permute were not integers. Instead you passed: {}'.format(indices_to_permute))
        else:
            indices_to_permute = indices_to_permute.astype(np.int64)

        if mcr_as_ratio:
            print('WARNING: You have set mcr_as_ratio as true. This part of the implementation is experimental and has not been checked for edge case correctness outside the dataset used in the Neurips paper.')

        #if not is_classification:
        #    raise Exception('This method is for classification ONLY')
        
        np.random.seed(seed)

        real_estimators_ = self.estimators_ # save (via pointers) the actual estimtors, we're going to mainpulate the trees
        
        """ Return the accuracy of the prediction of X compared to y. """

        acc_set_sur_and_truffle = []
        acc_set_sur_only = []
        acc_set_ref_model = []

        
        if restrict_trees_to is None:
            n_trees = len(self.estimators_)
        else:
            n_trees = restrict_trees_to
 

        amount_left_set_by = []
        y = y_in

        # Setup e_switch
        if e_switch:
            if len(indices_to_permute) != 1:
                raise Exception('The e switch implementation only allows the consideration of one input feature at a time at the moment.')
            y = np.zeros(len(y_in)*(len(y_in)-1))
            num_times = 1
            X = X_in[0:0].copy()
            ylenm1 = len(y_in)-1

            X_perm = X_in[0:0].copy()
            for i in range(X_in.shape[0]):
                idxs = np.ones(len(y_in),dtype=np.bool)
                idxs[i] = False
                y[i*ylenm1:i*ylenm1+ylenm1] = y_in[idxs]
                xtmp0 = np.delete(X_in,i,axis=0)    
                xtmp = xtmp0.copy()
                for j in range(xtmp.shape[0]):
                    xtmp[j,indices_to_permute[0]] = X_in[i,indices_to_permute[0]]
                    
                X = np.vstack( (X, xtmp0 ) )
                X_perm = np.vstack( (X_perm, xtmp) )
            
            #print('p')
        #perm_gener = permutations(range(X_in.shape[0]))
        
        
        n_samples = len(y)

        for i_num_times in range(num_times):
            # Make a copy of X and permute
            
            if not e_switch:
                X_perm = X_in.copy()
                X = X_in
        
                for i_permidx in indices_to_permute:
                    np.random.shuffle(X_perm[:, i_permidx])


                    #inplace_permute_or_random_sample_with_replacement( X_perm, i, permute = permute )
            
            # shuffle_idxs = next(perm_gener)
            # X_perm = X_in.copy()
            # X_perm[:,indices_to_permute[0]] = (X_perm[:,indices_to_permute[0]])[np.asarray(shuffle_idxs)]

            #forest_scorer_reference_model = lambda x: np.mean( (self.predict(x)-y_in)**2 )

            #acc_set_ref_model.append(forest_scorer_reference_model(X_perm) - forest_scorer_reference_model(X_in) )
            #print(np.sum(X_in))
            #print('{} - {}'.format(forest_scorer_reference_model(X_perm),forest_scorer_reference_model(X_in)))
            #return


            # for each tree we will predict all samples and store them here
            per_fplus_tree_preds = np.ones([n_trees, n_samples])*-9999
            per_ref_tree_preds = np.ones([n_trees, n_samples])*-9999


            def collate_parallel( eidx ):
                per_ref_tree_preds[eidx,:] = self.predict_tree(self.estimators_[eidx],X)
                per_fplus_tree_preds[eidx,:] = self.predict_vim_tree(self.estimators_[eidx],X_perm, indices_to_permute, mcr_type=mcr_type)

            if self.n_jobs is None or self.n_jobs == 1:

                for eidx in range(n_trees):
                    per_ref_tree_preds[eidx,:] = self.predict_tree(self.estimators_[eidx],X)
                    per_fplus_tree_preds[eidx,:] = self.predict_vim_tree(self.estimators_[eidx],X_perm, indices_to_permute, mcr_type=mcr_type)
            
            else:
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose, **_joblib_parallel_args(require="sharedmem"))(delayed(collate_parallel)(eidx) for eidx in range(n_trees))
            
            
            # for each tree make predictions for all samples using f+
            for eidx in range(n_trees):
                per_ref_tree_preds[eidx,:] = self.predict_tree(self.estimators_[eidx],X)

                per_fplus_tree_preds[eidx,:] = self.predict_vim_tree(self.estimators_[eidx],X_perm, indices_to_permute, mcr_type=mcr_type)
                                                                      # predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1)
            
            # turn the predictions into either 1-0 loss or squared error with regard to the truth
            if is_classification:
                # acc(f) - acc(f+), if this is high, we've seen lots of damage by the surrogates, if it is low little damage
                per_tree_diff_in_loss_ref_tree_vs_fplus_tree = (
                    np.mean( per_ref_tree_preds == np.tile(y,(n_trees,1)), axis = 1 ) - np.mean( per_fplus_tree_preds == np.tile(y,(n_trees,1)), axis = 1 )                       
                )
            else:
                # se(f+) - se(f), if this is high, we've seen lots of damage by the surrogates, if it is low little damage
                 
                if self.get_params()['criterion'] == 'mse':
                    per_tree_diff_in_loss_ref_tree_vs_fplus_tree = np.mean( (per_fplus_tree_preds - np.tile(y,(n_trees,1)))**2, axis = 1 ) - np.mean( (per_ref_tree_preds - np.tile(y,(n_trees,1)))**2, axis = 1 )
                else:
                    per_tree_diff_in_loss_ref_tree_vs_fplus_tree = np.mean( np.abs(per_fplus_tree_preds - np.tile(y,(n_trees,1))), axis = 1 ) - np.mean( np.abs(per_ref_tree_preds - np.tile(y,(n_trees,1))), axis = 1 )
            
            


            # Build up a new forest (the one we will take the MR of to get MCR+/-)
            # We do this by storing trees by index (into the trees used by the reference model)

            #############
            # BEING BUILD NEW FOREST

            new_trees_indexes = []
            for i_ntrees in range(n_trees):
                if enable_Tplus_transform:
                    if mcr_type > 0:
                        # MCR+
                        # per_tree_diff_in_loss_ref_tree_vs_fplus_tree
                        # contains the damage to the performance measure done by using f+ in excess of that that already existed in f 
                        
                        # for this tree get the set of trees that have accuacy equiviliency
                        indexs_of_eq_forests = self.forest_equivilents[i_ntrees]
                        # look over these and find the tree (index) that has the most loss of accuaracy
                        worst_tree = indexs_of_eq_forests[ np.argmax( per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] ) ]

                        # Use the found tree in the new forest
                        new_trees_indexes.append(worst_tree)
                        
                    else:
                        indexs_of_eq_forests = self.forest_equivilents[i_ntrees]
                        best_tree = indexs_of_eq_forests[ np.argmin( per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] ) ]

                        new_trees_indexes.append(best_tree)
                else:
                    new_trees_indexes.append(i_ntrees)

                   
            if is_classification:
                forest_scorer = lambda x: np.mean(self.predict_vim(x,indices_to_permute,mcr_type)==y)
            else:
                if self.get_params()['criterion'] == 'mse':
                    forest_scorer = lambda x: np.mean( (self.predict_vim(x,indices_to_permute,mcr_type)-y)**2 )
                else:
                    forest_scorer = lambda x: np.mean( np.abs(self.predict_vim(x,indices_to_permute,mcr_type)-y) )

            if is_classification:
                forest_scorer_reference_model = lambda x: np.mean(self.predict(x)==y)
            else:
                if self.get_params()['criterion'] == 'mse':
                    forest_scorer_reference_model = lambda x: np.mean( (self.predict(x)-y)**2 )
                else:
                    forest_scorer_reference_model = lambda x: np.mean( np.abs(self.predict(x)-y) )

            ref_forest_orig_data_score = forest_scorer(X)

            #acc_no_per_surrogates_VK = forest_scorer(X)
            if debug:
                acc_with_per_surrogates_VK = forest_scorer(X_perm)

            ##############################################
            # CHANGE OUR FOREST TO THE NEW FOREST
            ##############################################
            self.estimators_ = np.asarray(real_estimators_)[new_trees_indexes]

            # Check if we are still in the Rashomon set
            new_forest_orig_data_score = forest_scorer(X)
            
            rashomon_set_error = np.abs(ref_forest_orig_data_score - new_forest_orig_data_score)
            amount_left_set_by.append(rashomon_set_error)
            #if  rashomon_set_error > 0.00001:
            #    print('WARNING: Left the Rashomon set. Original acc/squared error: {}, After: {}, Difference: {}'.format(ref_forest_orig_data_score,new_forest_orig_data_score,rashomon_set_error))

            
            new_forest_orig_data_score = forest_scorer(X)
            new_forest_perm_data_score = forest_scorer(X_perm)

            
            ##############################################
            # SET THE FOREST BACK TO THE REFERENCE FORESET
            ##############################################
            self.estimators_ = real_estimators_
     
            #MDA for classifier Mean increase in Squared Error regressors
            if is_classification:

                if mcr_as_ratio:
                    sur_and_truffle = new_forest_perm_data_score/new_forest_orig_data_score
                else:
                    sur_and_truffle = new_forest_orig_data_score - new_forest_perm_data_score
                
                acc_set_sur_and_truffle.append(sur_and_truffle)

                if debug:
                    acc_set_sur_only.append(ref_forest_orig_data_score - acc_with_per_surrogates_VK)
                    acc_set_ref_model.append(forest_scorer_reference_model(X) - forest_scorer_reference_model(X_perm) )
            else:
                if mcr_as_ratio:
                    sur_and_truffle = new_forest_perm_data_score / new_forest_orig_data_score
                else:
                    sur_and_truffle = new_forest_perm_data_score - new_forest_orig_data_score
                acc_set_sur_and_truffle.append(sur_and_truffle)
                
                if debug:
                    acc_set_sur_only.append(acc_with_per_surrogates_VK - ref_forest_orig_data_score)
                    acc_set_ref_model.append(forest_scorer_reference_model(X_perm) - forest_scorer_reference_model(X) )

            
            # print('T0(X_perm) = {}'.format( np.mean((self.estimators_[0].predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 
            # print('T1(X_perm) = {}'.format( np.mean((self.estimators_[1].predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 

            # print('T[0,1](X_perm) = {}'.format( np.mean((self.predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 
            # self.estimators_ = np.asarray(real_estimators_)[new_trees_indexes]
            # print('T{}(X_perm) = {}'.format( new_trees_indexes, np.mean((self.predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 
            # self.estimators_ = real_estimators_
        
        #if (np.asarray(acc_set_sur_and_truffle) +  < 0).any() or (np.asarray(acc_set_sur_only)<0).any() or (np.asarray(acc_set_ref_model)<0).any():
            
        #    raise Exception('SHOULD NOT HAPPEN lsdajf23lkjd')


        if debug:

            return np.mean(acc_set_sur_and_truffle), np.mean(acc_set_sur_only),np.mean(acc_set_ref_model), 'min: {:.5f}, mean: {:.5f}, max: {:.5f}'.format(np.min(amount_left_set_by), np.mean(amount_left_set_by), np.max(amount_left_set_by)) #new

        return np.mean(acc_set_sur_and_truffle)


    def plot_mcr(self,X_in, y_in, feature_names = None, feature_groups_of_interest = 'all individual features', num_times = 100, show_fig = True):

        """
        Compute the required information for an MCR plot and optionally display the MCR plot.

        Parameters
        ----------
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
        """   
            
        if isinstance(X_in, pd.DataFrame):
            X = X_in.values
            if feature_names is None:
                feature_names = X_in.columns.tolist()
        else:
            X = X_in
            if feature_names is None:
                feature_names = ['f_{}'.format(i) for i in range(X.shape[1])]

        
        if isinstance(y_in, pd.DataFrame): 
            y = y_in.values
        else:
            y = y_in

        
        results = []
        if isinstance(feature_groups_of_interest, str):
            if feature_groups_of_interest == 'all individual features':
                groups_of_indicies_to_permute = [[x] for x in range(len(feature_names))]
            else:
                raise Exception('feature_groups_of_interest incorrectly specified. If not specifying to use all individual features via "all individual features" you must pass a numpy array of numpy arrays. See the documentation on github.')
        elif not isinstance(feature_groups_of_interest, np.array):
            raise Exception('feature_groups_of_interest incorrectly specified. If not specifying to use all individual features via "all individual features" you must pass a numpy array of numpy arrays. See the documentation on github.')
        
        
        # New MCR+ perm imp
        for gp in groups_of_indicies_to_permute:
            rn = self.mcr(X,y, np.asarray(gp) ,  num_times = num_times, mcr_type = 1)
            results.append([','.join([feature_names[x] for x in gp]), 'RF-MCR+', rn])


        # New MCR- perm imp
        for gp in groups_of_indicies_to_permute:
            rn = self.mcr(X,y, np.asarray(gp) ,  num_times = num_times,  mcr_type = -1)
            results.append([','.join([feature_names[x] for x in gp]), 'RF-MCR-', rn])

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
        if show_fig:
            plt.show()

        return rf_results2

# GAVIN TODO: Integrate better
    def mcr_extra(self, X_in, y_in, indices_to_permute, e_switch = False, 
                                    num_times = 100, debug = False, mcr_type = 1, restrict_trees_to = None, mcr_as_ratio = False, seed = 13111985 ):
        
        is_classification = is_classifier(self)

        #if not is_classification:
        #    raise Exception('This method is for classification ONLY')
        
        np.random.seed(seed)

        real_estimators_ = self.estimators_ # save (via pointers) the actual estimtors, we're going to mainpulate the trees
        
        """ Return the accuracy of the prediction of X compared to y. """

        acc_set_sur_and_truffle = []
        acc_set_sur_only = []
        acc_set_ref_model = []

        
        if restrict_trees_to is None:
            n_trees = len(self.estimators_)
        else:
            n_trees = restrict_trees_to
 

        amount_left_set_by = []
        y = y_in

        # Setup e_switch
        if e_switch:
            if len(indices_to_permute) != 1:
                raise Exception('The e switch implementation only allows the consideration of one input feature at a time at the moment.')
            y = np.zeros(len(y_in)*(len(y_in)-1))
            num_times = 1
            X = X_in[0:0].copy()
            ylenm1 = len(y_in)-1

            X_perm = X_in[0:0].copy()
            for i in range(X_in.shape[0]):
                idxs = np.ones(len(y_in),dtype=np.bool)
                idxs[i] = False
                y[i*ylenm1:i*ylenm1+ylenm1] = y_in[idxs]
                xtmp0 = np.delete(X_in,i,axis=0)    
                xtmp = xtmp0.copy()
                for j in range(xtmp.shape[0]):
                    xtmp[j,indices_to_permute[0]] = X_in[i,indices_to_permute[0]]
                    
                X = np.vstack( (X, xtmp0 ) )
                X_perm = np.vstack( (X_perm, xtmp) )
            
            #print('p')
        #perm_gener = permutations(range(X_in.shape[0]))
        
        
        n_samples = len(y)

        for i_num_times in range(num_times):
            # Make a copy of X and permute
            
            if not e_switch:
                X_perm = X_in.copy()
                X = X_in
        
                for i_permidx in indices_to_permute:
                    np.random.shuffle(X_perm[:, i_permidx])


                    #inplace_permute_or_random_sample_with_replacement( X_perm, i, permute = permute )
            
            # shuffle_idxs = next(perm_gener)
            # X_perm = X_in.copy()
            # X_perm[:,indices_to_permute[0]] = (X_perm[:,indices_to_permute[0]])[np.asarray(shuffle_idxs)]

            #forest_scorer_reference_model = lambda x: np.mean( (self.predict(x)-y_in)**2 )

            #acc_set_ref_model.append(forest_scorer_reference_model(X_perm) - forest_scorer_reference_model(X_in) )
            #print(np.sum(X_in))
            #print('{} - {}'.format(forest_scorer_reference_model(X_perm),forest_scorer_reference_model(X_in)))
            #return


            # for each tree we will predict all samples and store them here
            per_fplus_tree_preds = np.ones([n_trees, n_samples])*-9999
            per_ref_tree_preds = np.ones([n_trees, n_samples])*-9999
            
            # for each tree make predictions for all samples using f+
            for eidx in range(n_trees):
                per_ref_tree_preds[eidx,:] = self.predict_tree(self.estimators_[eidx],X)

                per_fplus_tree_preds[eidx,:] = self.predict_vim_tree(self.estimators_[eidx],X_perm, indices_to_permute, mcr_type=mcr_type)
                                                                      # predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1)
            
            # turn the predictions into either 1-0 loss or squared error with regard to the truth
            if is_classification:
                # acc(f) - acc(f+), if this is high, we've seen lots of damage by the surrogates, if it is low little damage
                per_tree_diff_in_loss_ref_tree_vs_fplus_tree = (
                    np.mean( per_ref_tree_preds == np.tile(y,(n_trees,1)), axis = 1 ) - np.mean( per_fplus_tree_preds == np.tile(y,(n_trees,1)), axis = 1 )                       
                )
            else:
                # se(f+) - se(f), if this is high, we've seen lots of damage by the surrogates, if it is low little damage
                 
                if self.get_params()['criterion'] == 'mse':
                    per_tree_diff_in_loss_ref_tree_vs_fplus_tree = np.mean( (per_fplus_tree_preds - np.tile(y,(n_trees,1)))**2, axis = 1 ) - np.mean( (per_ref_tree_preds - np.tile(y,(n_trees,1)))**2, axis = 1 )
                else:
                    per_tree_diff_in_loss_ref_tree_vs_fplus_tree = np.mean( np.abs(per_fplus_tree_preds - np.tile(y,(n_trees,1))), axis = 1 ) - np.mean( np.abs(per_ref_tree_preds - np.tile(y,(n_trees,1))), axis = 1 )
            
            


            # Build up a new forest (the one we will take the MR of to get MCR+/-)
            # We do this by storing trees by index (into the trees used by the reference model)

            #############
            # BEING BUILD NEW FOREST

            new_trees_indexes = []
            for i_ntrees in range(n_trees):
                if mcr_type > 0:
                    # MCR+
                    # per_tree_diff_in_loss_ref_tree_vs_fplus_tree
                    # contains the damage to the performance measure done by using f+ in excess of that that already existed in f 
                    
                    # for this tree get the set of trees that have accuacy equiviliency
                    indexs_of_eq_forests = self.forest_equivilents[i_ntrees]
                    indexs_of_eq_forests = np.asarray([x for x in indexs_of_eq_forests if x < n_trees])
                    # look over these and find the tree (index) that has the most loss of accuaracy
                    worst_tree = indexs_of_eq_forests[ np.argmax( per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] ) ]

                    # Use the found tree in the new forest
                    new_trees_indexes.append(worst_tree)
                      
                else:
                    indexs_of_eq_forests = self.forest_equivilents[i_ntrees]
                    indexs_of_eq_forests = np.asarray([x for x in indexs_of_eq_forests if x < n_trees])
                    best_tree = indexs_of_eq_forests[ np.argmin( per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] ) ]

                    new_trees_indexes.append(best_tree)

                   
            if is_classification:
                forest_scorer = lambda x: np.mean(self.predict_vim(x,indices_to_permute,mcr_type)==y)
            else:
                if self.get_params()['criterion'] == 'mse':
                    forest_scorer = lambda x: np.mean( (self.predict_vim(x,indices_to_permute,mcr_type)-y)**2 )
                else:
                    forest_scorer = lambda x: np.mean( np.abs(self.predict_vim(x,indices_to_permute,mcr_type)-y) )

            if is_classification:
                forest_scorer_reference_model = lambda x: np.mean(self.predict(x)==y)
            else:
                if self.get_params()['criterion'] == 'mse':
                    forest_scorer_reference_model = lambda x: np.mean( (self.predict(x)-y)**2 )
                else:
                    forest_scorer_reference_model = lambda x: np.mean( np.abs(self.predict(x)-y) )

            ref_forest_orig_data_score = forest_scorer(X)

            #acc_no_per_surrogates_VK = forest_scorer(X)
            if debug:
                acc_with_per_surrogates_VK = forest_scorer(X_perm)

            ##############################################
            # CHANGE OUR FOREST TO THE NEW FOREST
            ##############################################
            self.estimators_ = np.asarray(real_estimators_)[new_trees_indexes]

            # Check if we are still in the Rashomon set
            new_forest_orig_data_score = forest_scorer(X)
            
            rashomon_set_error = np.abs(ref_forest_orig_data_score - new_forest_orig_data_score)
            amount_left_set_by.append(rashomon_set_error)
            #if  rashomon_set_error > 0.00001:
            #    print('WARNING: Left the Rashomon set. Original acc/squared error: {}, After: {}, Difference: {}'.format(ref_forest_orig_data_score,new_forest_orig_data_score,rashomon_set_error))

            
            new_forest_orig_data_score = forest_scorer(X)
            new_forest_perm_data_score = forest_scorer(X_perm)

            
            ##############################################
            # SET THE FOREST BACK TO THE REFERENCE FORESET
            ##############################################
            self.estimators_ = real_estimators_
     
            #MDA for classifier Mean increase in Squared Error regressors
            if is_classification:
                
                if mcr_as_ratio:
                    sur_and_truffle = new_forest_perm_data_score/new_forest_orig_data_score
                    raise Exception('MCR as ratio for classification is likely currently computed incorrectly.')
                else:
                    sur_and_truffle = new_forest_orig_data_score - new_forest_perm_data_score
                
                acc_set_sur_and_truffle.append(sur_and_truffle)

                if debug:
                    acc_set_sur_only.append(ref_forest_orig_data_score - acc_with_per_surrogates_VK)
                    acc_set_ref_model.append(forest_scorer_reference_model(X) - forest_scorer_reference_model(X_perm) )
            else:
                if mcr_as_ratio:
                    sur_and_truffle = new_forest_perm_data_score / new_forest_orig_data_score
                else:
                    sur_and_truffle = new_forest_perm_data_score - new_forest_orig_data_score
                acc_set_sur_and_truffle.append(sur_and_truffle)
                
                if debug:
                    acc_set_sur_only.append(acc_with_per_surrogates_VK - ref_forest_orig_data_score)
                    acc_set_ref_model.append(forest_scorer_reference_model(X_perm) - forest_scorer_reference_model(X) )

            
            # print('T0(X_perm) = {}'.format( np.mean((self.estimators_[0].predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 
            # print('T1(X_perm) = {}'.format( np.mean((self.estimators_[1].predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 

            # print('T[0,1](X_perm) = {}'.format( np.mean((self.predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 
            # self.estimators_ = np.asarray(real_estimators_)[new_trees_indexes]
            # print('T{}(X_perm) = {}'.format( new_trees_indexes, np.mean((self.predict_vim(X_perm, np.asarray([indices_to_permute[0]]), -1) - y)**2)) ) 
            # self.estimators_ = real_estimators_
        
        #if (np.asarray(acc_set_sur_and_truffle) +  < 0).any() or (np.asarray(acc_set_sur_only)<0).any() or (np.asarray(acc_set_ref_model)<0).any():
            
        #    raise Exception('SHOULD NOT HAPPEN lsdajf23lkjd')


        if debug:

            return np.mean(acc_set_sur_and_truffle), np.mean(acc_set_sur_only),np.mean(acc_set_ref_model), 'min: {:.5f}, mean: {:.5f}, max: {:.5f}'.format(np.min(amount_left_set_by), np.mean(amount_left_set_by), np.max(amount_left_set_by)) #new

        return np.mean(acc_set_sur_and_truffle)


    # GAVIN TODO: Integrate better
    def permutation_importance_orig(self, X_in, y_in, indices_to_permute, permute, pre_permutated = False, num_times = 100, debug = False):
        
        is_classification = is_classifier(self)
        
        """ Return the accuracy of the prediction of X compared to y. """
        np.random.seed(13111985)
        if is_classification:
            base_score = self.score(X_in,y_in)
            card_y = len(np.unique(y_in))
        else:
            if self.get_params()['criterion'] == 'mse':
                base_score = mean_squared_error(y_in, self.predict(X_in))
            elif self.get_params()['criterion'] == 'mae':
                base_score = mean_absolute_error(y_in, self.predict(X_in))
            else:
                raise Exception('Unsupported criterion: {}'.format(self.get_params()['criterion']))

        acc_set = []
        n_samples = len(y_in)
        
        n_trees = len(self.estimators_)
       
       # print('====d=========================')
        for i in range(num_times):
            #print('==========ud==================: {}'.format(indices_to_permute))
            X = X_in.copy()
            y = y_in.copy()
            if not pre_permutated:
                for i in indices_to_permute:
                    #print('==w===========================')
                    inplace_permute_or_random_sample_with_replacement( X, i, permute = permute )
                    #np.random.shuffle(X[:, i])



            if is_classification:
                # MDA
                acc_set.append(base_score- self.score(X,y))
                
            else:
                # Here we return Mean increase in Error (MIE)
                if self.get_params()['criterion'] == 'mse':
                    acc_set.append(mean_squared_error(y, self.predict(X)) - base_score)
                elif self.get_params()['criterion'] == 'mae':
                    acc_set.append(mean_absolute_error(y, self.predict(X)) - base_score)
                else:
                    raise Exception('Unsupported criterion: {}'.format(self.get_params()['criterion']))
                
        
        mean_performace = np.mean(acc_set) #calculate the average accuracy
        
        return mean_performace #new



    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           **_joblib_parallel_args(prefer="threads"))(
            delayed(tree.apply)(X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              **_joblib_parallel_args(prefer='threads'))(
            delayed(tree.decision_path)(X, check_input=False)
            for tree in self.estimators_)

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """

        if is_classifier(self):
            if np.sum(np.asarray(y) == 0) + np.sum(np.asarray(y) == 1) != len(y):
                raise Exception('For classification, output labels must be either 0 or 1.') 
        
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)


        # GAVIN VERIFY BUILD OF TREES
        from sklearn.metrics import accuracy_score
        for tidx, t in enumerate(self.estimators_):
            if is_classifier(self):
                a = accuracy_score(y, t.predict(X))
            else:
                a = mean_squared_error(y, t.predict(X))
            
            for i in range( X.shape[1] ):
                # if tidx==3 and i == 2:
                #     print('p')
                if is_classifier(self):
                    b = accuracy_score(y, t.predict_vim(X,np.asarray([i], dtype=np.int64), 1))
                else:
                    b = mean_squared_error(y, t.predict_vim(X,np.asarray([i], dtype=np.int64), 1))
                
                #b = mean_squared_error(y[5], t.predict_vim(X[5,:].reshape(1,-1),np.asarray([i]), 1))
                #c = mean_squared_error(y, t.predict_vim(X,np.asarray([i], dtype=np.int64), -1))
                if a != b:
                    print('np.asarray([i], dtype=np.int64): {}'.format(np.asarray([i], dtype=np.int64)))
                    print('Bootstrap: {}'.format(self.bootstrap))
                    print('a: {}'.format(a))
                    print('b: {}'.format(b))
                    raise Exception('MAJOR SURROGATE ERROR WITH MCR+. Or you forgot to set bootstrap = False.')
                #if a != c:
                #    raise Exception('MAJOR SURROGATE ERROR WITH MCR-. Or you forgot to set bootstrap = False.')

        # GAVIN

        self.forest_equivilents = []

        if self.performance_equivilence:
            self.forest_scores = []

            if is_classifier(self):
                for e in self.estimators_:
                    self.forest_scores.append( e.score(X,y) )
            else:
                for e in self.estimators_:
                    self.forest_scores.append( mean_squared_error(y,e.predict(X)) )      
        
            #print('ACC DIVERSITY: {}'.format(self.forest_scores))
            # learn the tree equivliance (if any)
            # TODO: is there a computationally better way of doing this??

            
            for i in range(self.n_estimators):
                eq_set = []
                for j in range(self.n_estimators):
                
                    if np.abs(self.forest_scores[i] - self.forest_scores[j]) <= self.mcr_tree_equivilient_tol:
                        eq_set.append(j)

                self.forest_equivilents.append( eq_set )
        else:
            # prediction equvilence
            forest_preds = []
            for i in range(self.n_estimators):
                forest_preds.append( self.estimators_[i].predict(X) )

            for i in range(self.n_estimators):
                eq_set = []
                ie = forest_preds[i]
                for j in range(self.n_estimators):
                    if is_classifier(self):
                        if (1-np.mean(np.abs(ie == forest_preds[j]))) <= self.mcr_tree_equivilient_tol: 
                            eq_set.append(j)
                            #if i != j:
                            #    print('FOUND PREDICTION EQUIVLIACNE (Classification)')
                    else:
                        if mean_squared_error(ie, forest_preds[j]) <= self.mcr_tree_equivilient_tol:
                        #if (ie == forest_preds[j]).all(): 
                            eq_set.append(j)
                            #if i != j:
                            #    print('FOUND PREDICTION EQUIVLIACNE (Regression)')
                self.forest_equivilents.append( eq_set )
        # END GAVIN

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
    
    @abstractmethod
    def predict_vim(self, X,indices_to_permute, mcr_type):
        """stuff"""

    # @abstractmethod
    # def predict_proba_vim(self, X,permuted_vars, mcr_type):
    #     """stuff"""

    @abstractmethod
    def score(self, X, y, sample_weight=None):
        """stuff"""

    @abstractmethod
    def predict(self, X):
        """
        Make a prediction
        """

    @abstractmethod
    def _set_oob_score(self, X, y):
        """
        Calculate out of bag predictions and score."""

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   **_joblib_parallel_args(prefer='threads'))(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 max_samples=None,mcr_tree_equivilient_tol = 0.0001, performance_equivilence = True):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,mcr_tree_equivilient_tol=mcr_tree_equivilient_tol,
            performance_equivilence = performance_equivilence)






    def _set_oob_score(self, X, y):
        """
        Compute out-of-bag score."""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = [np.zeros((n_samples, n_classes_[k]))
                       for k in range(self.n_outputs_)]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples, n_samples_bootstrap)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = \
                np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample".'
                                     'Given "%s".'
                                     % self.class_weight)
                if self.warm_start:
                    warn('class_weight presets "balanced" or '
                         '"balanced_subsample" are '
                         'not recommended for warm_start if the fitted data '
                         'differs from the full dataset. In order to use '
                         '"balanced" weights, use compute_class_weight '
                         '("balanced", classes, y). In place of y you can use '
                         'a large enough sample of the full training set '
                         'target to properly estimate the class frequency '
                         'distributions. Pass the resulting weights as the '
                         'class_weight parameter.')

            if (self.class_weight != 'balanced_subsample' or
                    not self.bootstrap):
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight,
                                                              y_original)

        return y, expanded_class_weight

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions


    def predict_vim(self, X, permuated_vars, mcr_type):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba_vim(X,permuated_vars, mcr_type)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions



    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba,
                                            lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_proba_vim(self, X,permuted_vars, mcr_type):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
    
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict_proba_vim, (X,permuted_vars, mcr_type), all_proba,
                                            lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba



    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None,
                 mcr_tree_equivilient_tol = 0.00001, performance_equivilence = True):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            mcr_tree_equivilient_tol = mcr_tree_equivilient_tol,performance_equivilence= performance_equivilence)


        # GAVIN TODO: Integrate better
    # def mcr(self, X_in, y_in, indices_to_permute, permute, pre_permutated = False, 
    #                                 num_times = 100, debug = False, mcr_type = 1, restrict_trees_to = None, mcr_as_ratio = False, seed = 13111985 ):
        
    #     is_classification = is_classifier(self)

    #     if is_classification:
    #         raise Exception('This method is for regression ONLY')

    #     if not seed is None:
    #         np.random.seed(seed)
    #     """ Return the accuracy of the prediction of X compared to y. """
        
    #     if is_classification:
    #         base_score = self.score(X_in,y_in)
    #         card_y = len(np.unique(y_in))
    #     else:
    #         if self.get_params()['criterion'] == 'mse':
    #             base_score = mean_squared_error(y_in, self.predict(X_in))
    #         elif self.get_params()['criterion'] == 'mae':
    #             base_score = mean_absolute_error(y_in, self.predict(X_in))
    #         else:
    #             raise Exception('Unsupported criterion: {}'.format(self.get_params()['criterion']))

    #     acc_set = []
    #     n_samples = len(y_in)
    #     if restrict_trees_to is None:
    #         n_trees = len(self.estimators_)
    #     else:
    #         n_trees = restrict_trees_to
    #    # print('====d=========================')
    #     for i in range(num_times):
    #         #print('==========ud==================: {}'.format(indices_to_permute))
    #         X = X_in.copy()
    #         y = y_in.copy()
    #         if not pre_permutated:
    #             for i in indices_to_permute:
    #                 #print('==w===========================')
    #                 inplace_permute_or_random_sample_with_replacement( X, i, permute = permute )
    #                 #np.random.shuffle(X[:, i])

            
    #         y_predict_for_each_tree = np.ones([n_trees, n_samples])*-9999
    #         for eidx in range(n_trees):
    #             y_predict_for_each_tree[eidx,:] = self.estimators_[eidx].predict_vim(X, indices_to_permute, mcr_type=mcr_type)

    #         # Compute the per tree prediction accuracys

    #         # Compute (by averaging across all samples) the tree MDA
    #         # per_tree_base_minus_acc is now a 1D set of averages, one per tree. 
    #         # the average of these would be the overall MDA

    #         # we would normally convert to MDA or error gain here, but we might want either
    #         # a loss or a ratio, so we just make sure max is more damage and fix the exact
    #         # reported measure later
  
            
    #         #per_tree_base_predict_performace = np.mean( (y_predict_for_each_tree - np.tile(y,(n_trees,1)))**2, axis = 1 ) - base_score

    #         if self.get_params()['criterion'] == 'mse':
    #             per_tree_base_predict_performace = np.mean( (y_predict_for_each_tree - np.tile(y,(n_trees,1)))**2, axis = 1 ) 
    #         elif self.get_params()['criterion'] == 'mae':
    #             per_tree_base_predict_performace = np.mean( np.abs(y_predict_for_each_tree - np.tile(y,(n_trees,1))), axis = 1 ) 
    #         else:
    #             raise Exception('Unsupported criterion: {}'.format(self.get_params()['criterion']))

    #         # Adjust to consider equvilient trees
    #         if debug:
    #             print("Per tree MDA:")
    #         for i in range(n_trees):
    #             if mcr_type > 0:
    #                 per_tree_base_predict_performace[i] = max( per_tree_base_predict_performace[ [x for x in self.forest_equivilents[i] if x < n_trees] ] )
    #             else:
    #                 per_tree_base_predict_performace[i] = min( per_tree_base_predict_performace[ [x for x in self.forest_equivilents[i] if x < n_trees] ] )
    #             if debug:
    #                 print(per_tree_base_predict_performace[i]) 
            
    #         if not mcr_as_ratio:
    #             # Here we return Mean increase in Error (MIE)
    #             per_tree_base_predict_performace = per_tree_base_predict_performace - base_score
    #         else:
    #             per_tree_base_predict_performace = per_tree_base_predict_performace / base_score

        
    #         acc_set.append(np.mean(per_tree_base_predict_performace))
            
    #     mean_accuracy = np.mean(acc_set) #calculate the average accuracy over all repeats
            
    #     #print('RF New Perm: base:{}. MDA: {}'.format(base_score,mean_accuracy)) #new
    #     return mean_accuracy #new


    def predict_vim(self, X, permuted_indices, mcr_type):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict_vim_from_parallel_fn, (X,permuted_indices,mcr_type), [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat


    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat



    def _set_oob_score(self, X, y):
        """
        Compute out-of-bag scores."""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, self.max_samples
        )

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples, n_samples_bootstrap)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples, ))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        grid = np.asarray(grid, dtype=DTYPE, order='C')
        averaged_predictions = np.zeros(shape=grid.shape[0],
                                        dtype=np.float64, order='C')

        for tree in self.estimators_:
            # Note: we don't sum in parallel because the GIL isn't released in
            # the fast method.
            tree.tree_.compute_partial_dependence(
                grid, target_features, averaged_predictions)
        # Average over the forest
        averaged_predictions /= len(self.estimators_)

        return averaged_predictions

class RandomForestClassifier(ForestClassifier):
    """
    A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True`, otherwise the whole dataset is used to build
    each tree.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.


    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

        .. versionadded:: 0.22

    Attributes
    ----------
    base_estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    See Also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 mcr_tree_equivilient_tol = 0.00001, performance_equivilence = True, spoof_as_sklearn = False):
        super().__init__(
            base_estimator=DecisionTreeClassifier(spoof_as_sklearn=spoof_as_sklearn),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
            mcr_tree_equivilient_tol = mcr_tree_equivilient_tol, performance_equivilence = performance_equivilence)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.spoof_as_sklearn = spoof_as_sklearn
        

        if spoof_as_sklearn:
            from sklearn.ensemble import RandomForestClassifier as sklrfc
            self.__class__ = sklrfc

        if ccp_alpha != 0:
            raise Exception('Currently ccp_alpha is not supported.')

        if not max_leaf_nodes is None:
            raise Exception('For MCR, max_leaf_nodes is currently not supported.')

class RandomForestRegressor(ForestRegressor):
    """
    A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True`, otherwise the whole dataset is used to build
    each tree.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"mse", "mae"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

        .. versionadded:: 0.22

    Attributes
    ----------
    base_estimator_ : DecisionTreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    See Also
    --------
    DecisionTreeRegressor, ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    The default value ``max_features="auto"`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [1], whereas the former was more recently justified empirically in [2].

    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomForestRegressor(max_depth=2, random_state=0)
    >>> regr.fit(X, y)
    RandomForestRegressor(...)
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-8.32987858]
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None,
                 mcr_tree_equivilient_tol = 0.00001, performance_equivilence = True, spoof_as_sklearn = False):
        super().__init__(
            base_estimator=DecisionTreeRegressor(spoof_as_sklearn = spoof_as_sklearn),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            mcr_tree_equivilient_tol = mcr_tree_equivilient_tol, performance_equivilence = performance_equivilence)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

        if ccp_alpha != 0:
            raise Exception('Currently ccp_alpha is not supported.')

        if not max_leaf_nodes is None:
            raise Exception('For MCR, max_leaf_nodes is currently not supported.')

        self.spoof_as_sklearn = spoof_as_sklearn
        if spoof_as_sklearn:
            from sklearn.ensemble import RandomForestRegressor as sklrfr
            self.__class__ = sklrfr


# class ExtraTreesClassifier(ForestClassifier):
#     """
#     An extra-trees classifier.

#     This class implements a meta estimator that fits a number of
#     randomized decision trees (a.k.a. extra-trees) on various sub-samples
#     of the dataset and uses averaging to improve the predictive accuracy
#     and control over-fitting.

#     Read more in the :ref:`User Guide <forest>`.

#     Parameters
#     ----------
#     n_estimators : int, default=100
#         The number of trees in the forest.

#         .. versionchanged:: 0.22
#            The default value of ``n_estimators`` changed from 10 to 100
#            in 0.22.

#     criterion : {"gini", "entropy"}, default="gini"
#         The function to measure the quality of a split. Supported criteria are
#         "gini" for the Gini impurity and "entropy" for the information gain.

#     max_depth : int, default=None
#         The maximum depth of the tree. If None, then nodes are expanded until
#         all leaves are pure or until all leaves contain less than
#         min_samples_split samples.

#     min_samples_split : int or float, default=2
#         The minimum number of samples required to split an internal node:

#         - If int, then consider `min_samples_split` as the minimum number.
#         - If float, then `min_samples_split` is a fraction and
#           `ceil(min_samples_split * n_samples)` are the minimum
#           number of samples for each split.

#         .. versionchanged:: 0.18
#            Added float values for fractions.

#     min_samples_leaf : int or float, default=1
#         The minimum number of samples required to be at a leaf node.
#         A split point at any depth will only be considered if it leaves at
#         least ``min_samples_leaf`` training samples in each of the left and
#         right branches.  This may have the effect of smoothing the model,
#         especially in regression.

#         - If int, then consider `min_samples_leaf` as the minimum number.
#         - If float, then `min_samples_leaf` is a fraction and
#           `ceil(min_samples_leaf * n_samples)` are the minimum
#           number of samples for each node.

#         .. versionchanged:: 0.18
#            Added float values for fractions.

#     min_weight_fraction_leaf : float, default=0.0
#         The minimum weighted fraction of the sum total of weights (of all
#         the input samples) required to be at a leaf node. Samples have
#         equal weight when sample_weight is not provided.

#     max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
#         The number of features to consider when looking for the best split:

#         - If int, then consider `max_features` features at each split.
#         - If float, then `max_features` is a fraction and
#           `int(max_features * n_features)` features are considered at each
#           split.
#         - If "auto", then `max_features=sqrt(n_features)`.
#         - If "sqrt", then `max_features=sqrt(n_features)`.
#         - If "log2", then `max_features=log2(n_features)`.
#         - If None, then `max_features=n_features`.

#         Note: the search for a split does not stop until at least one
#         valid partition of the node samples is found, even if it requires to
#         effectively inspect more than ``max_features`` features.

#     max_leaf_nodes : int, default=None
#         Grow trees with ``max_leaf_nodes`` in best-first fashion.
#         Best nodes are defined as relative reduction in impurity.
#         If None then unlimited number of leaf nodes.

#     min_impurity_decrease : float, default=0.0
#         A node will be split if this split induces a decrease of the impurity
#         greater than or equal to this value.

#         The weighted impurity decrease equation is the following::

#             N_t / N * (impurity - N_t_R / N_t * right_impurity
#                                 - N_t_L / N_t * left_impurity)

#         where ``N`` is the total number of samples, ``N_t`` is the number of
#         samples at the current node, ``N_t_L`` is the number of samples in the
#         left child, and ``N_t_R`` is the number of samples in the right child.

#         ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
#         if ``sample_weight`` is passed.

#         .. versionadded:: 0.19

#     min_impurity_split : float, default=None
#         Threshold for early stopping in tree growth. A node will split
#         if its impurity is above the threshold, otherwise it is a leaf.

#         .. deprecated:: 0.19
#            ``min_impurity_split`` has been deprecated in favor of
#            ``min_impurity_decrease`` in 0.19. The default value of
#            ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
#            will be removed in 0.25. Use ``min_impurity_decrease`` instead.

#     bootstrap : bool, default=False
#         Whether bootstrap samples are used when building trees. If False, the
#         whole dataset is used to build each tree.

#     oob_score : bool, default=False
#         Whether to use out-of-bag samples to estimate
#         the generalization accuracy.

#     n_jobs : int, default=None
#         The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
#         :meth:`decision_path` and :meth:`apply` are all parallelized over the
#         trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
#         context. ``-1`` means using all processors. See :term:`Glossary
#         <n_jobs>` for more details.

#     random_state : int, RandomState, default=None
#         Controls 3 sources of randomness:

#         - the bootstrapping of the samples used when building trees
#           (if ``bootstrap=True``)
#         - the sampling of the features to consider when looking for the best
#           split at each node (if ``max_features < n_features``)
#         - the draw of the splits for each of the `max_features`

#         See :term:`Glossary <random_state>` for details.

#     verbose : int, default=0
#         Controls the verbosity when fitting and predicting.

#     warm_start : bool, default=False
#         When set to ``True``, reuse the solution of the previous call to fit
#         and add more estimators to the ensemble, otherwise, just fit a whole
#         new forest. See :term:`the Glossary <warm_start>`.

#     class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
#             default=None
#         Weights associated with classes in the form ``{class_label: weight}``.
#         If not given, all classes are supposed to have weight one. For
#         multi-output problems, a list of dicts can be provided in the same
#         order as the columns of y.

#         Note that for multioutput (including multilabel) weights should be
#         defined for each class of every column in its own dict. For example,
#         for four-class multilabel classification weights should be
#         [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
#         [{1:1}, {2:5}, {3:1}, {4:1}].

#         The "balanced" mode uses the values of y to automatically adjust
#         weights inversely proportional to class frequencies in the input data
#         as ``n_samples / (n_classes * np.bincount(y))``

#         The "balanced_subsample" mode is the same as "balanced" except that
#         weights are computed based on the bootstrap sample for every tree
#         grown.

#         For multi-output, the weights of each column of y will be multiplied.

#         Note that these weights will be multiplied with sample_weight (passed
#         through the fit method) if sample_weight is specified.

#     ccp_alpha : non-negative float, default=0.0
#         Complexity parameter used for Minimal Cost-Complexity Pruning. The
#         subtree with the largest cost complexity that is smaller than
#         ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
#         :ref:`minimal_cost_complexity_pruning` for details.

#         .. versionadded:: 0.22

#     max_samples : int or float, default=None
#         If bootstrap is True, the number of samples to draw from X
#         to train each base estimator.

#         - If None (default), then draw `X.shape[0]` samples.
#         - If int, then draw `max_samples` samples.
#         - If float, then draw `max_samples * X.shape[0]` samples. Thus,
#           `max_samples` should be in the interval `(0, 1)`.

#         .. versionadded:: 0.22

#     Attributes
#     ----------
#     base_estimator_ : ExtraTreesClassifier
#         The child estimator template used to create the collection of fitted
#         sub-estimators.

#     estimators_ : list of DecisionTreeClassifier
#         The collection of fitted sub-estimators.

#     classes_ : ndarray of shape (n_classes,) or a list of such arrays
#         The classes labels (single output problem), or a list of arrays of
#         class labels (multi-output problem).

#     n_classes_ : int or list
#         The number of classes (single output problem), or a list containing the
#         number of classes for each output (multi-output problem).

#     feature_importances_ : ndarray of shape (n_features,)
#         The impurity-based feature importances.
#         The higher, the more important the feature.
#         The importance of a feature is computed as the (normalized)
#         total reduction of the criterion brought by that feature.  It is also
#         known as the Gini importance.

#         Warning: impurity-based feature importances can be misleading for
#         high cardinality features (many unique values). See
#         :func:`sklearn.inspection.permutation_importance` as an alternative.

#     n_features_ : int
#         The number of features when ``fit`` is performed.

#     n_outputs_ : int
#         The number of outputs when ``fit`` is performed.

#     oob_score_ : float
#         Score of the training dataset obtained using an out-of-bag estimate.
#         This attribute exists only when ``oob_score`` is True.

#     oob_decision_function_ : ndarray of shape (n_samples, n_classes)
#         Decision function computed with out-of-bag estimate on the training
#         set. If n_estimators is small it might be possible that a data point
#         was never left out during the bootstrap. In this case,
#         `oob_decision_function_` might contain NaN. This attribute exists
#         only when ``oob_score`` is True.

#     See Also
#     --------
#     sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
#     RandomForestClassifier : Ensemble Classifier based on trees with optimal
#         splits.

#     Notes
#     -----
#     The default values for the parameters controlling the size of the trees
#     (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
#     unpruned trees which can potentially be very large on some data sets. To
#     reduce memory consumption, the complexity and size of the trees should be
#     controlled by setting those parameter values.

#     References
#     ----------
#     .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
#            trees", Machine Learning, 63(1), 3-42, 2006.

#     Examples
#     --------
#     >>> from sklearn.ensemble import ExtraTreesClassifier
#     >>> from sklearn.datasets import make_classification
#     >>> X, y = make_classification(n_features=4, random_state=0)
#     >>> clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
#     >>> clf.fit(X, y)
#     ExtraTreesClassifier(random_state=0)
#     >>> clf.predict([[0, 0, 0, 0]])
#     array([1])
#     """
#     @_deprecate_positional_args
#     def __init__(self,
#                  n_estimators=100, *,
#                  criterion="gini",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  bootstrap=False,
#                  oob_score=False,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False,
#                  class_weight=None,
#                  ccp_alpha=0.0,
#                  max_samples=None):
#         super().__init__(
#             base_estimator=ExtraTreeClassifier(),
#             n_estimators=n_estimators,
#             estimator_params=("criterion", "max_depth", "min_samples_split",
#                               "min_samples_leaf", "min_weight_fraction_leaf",
#                               "max_features", "max_leaf_nodes",
#                               "min_impurity_decrease", "min_impurity_split",
#                               "random_state", "ccp_alpha"),
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start,
#             class_weight=class_weight,
#             max_samples=max_samples)

#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_features = max_features
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.min_impurity_split = min_impurity_split
#         self.ccp_alpha = ccp_alpha


# class ExtraTreesRegressor(ForestRegressor):
#     """
#     An extra-trees regressor.

#     This class implements a meta estimator that fits a number of
#     randomized decision trees (a.k.a. extra-trees) on various sub-samples
#     of the dataset and uses averaging to improve the predictive accuracy
#     and control over-fitting.

#     Read more in the :ref:`User Guide <forest>`.

#     Parameters
#     ----------
#     n_estimators : int, default=100
#         The number of trees in the forest.

#         .. versionchanged:: 0.22
#            The default value of ``n_estimators`` changed from 10 to 100
#            in 0.22.

#     criterion : {"mse", "mae"}, default="mse"
#         The function to measure the quality of a split. Supported criteria
#         are "mse" for the mean squared error, which is equal to variance
#         reduction as feature selection criterion, and "mae" for the mean
#         absolute error.

#         .. versionadded:: 0.18
#            Mean Absolute Error (MAE) criterion.

#     max_depth : int, default=None
#         The maximum depth of the tree. If None, then nodes are expanded until
#         all leaves are pure or until all leaves contain less than
#         min_samples_split samples.

#     min_samples_split : int or float, default=2
#         The minimum number of samples required to split an internal node:

#         - If int, then consider `min_samples_split` as the minimum number.
#         - If float, then `min_samples_split` is a fraction and
#           `ceil(min_samples_split * n_samples)` are the minimum
#           number of samples for each split.

#         .. versionchanged:: 0.18
#            Added float values for fractions.

#     min_samples_leaf : int or float, default=1
#         The minimum number of samples required to be at a leaf node.
#         A split point at any depth will only be considered if it leaves at
#         least ``min_samples_leaf`` training samples in each of the left and
#         right branches.  This may have the effect of smoothing the model,
#         especially in regression.

#         - If int, then consider `min_samples_leaf` as the minimum number.
#         - If float, then `min_samples_leaf` is a fraction and
#           `ceil(min_samples_leaf * n_samples)` are the minimum
#           number of samples for each node.

#         .. versionchanged:: 0.18
#            Added float values for fractions.

#     min_weight_fraction_leaf : float, default=0.0
#         The minimum weighted fraction of the sum total of weights (of all
#         the input samples) required to be at a leaf node. Samples have
#         equal weight when sample_weight is not provided.

#     max_features : {"auto", "sqrt", "log2"} int or float, default="auto"
#         The number of features to consider when looking for the best split:

#         - If int, then consider `max_features` features at each split.
#         - If float, then `max_features` is a fraction and
#           `int(max_features * n_features)` features are considered at each
#           split.
#         - If "auto", then `max_features=n_features`.
#         - If "sqrt", then `max_features=sqrt(n_features)`.
#         - If "log2", then `max_features=log2(n_features)`.
#         - If None, then `max_features=n_features`.

#         Note: the search for a split does not stop until at least one
#         valid partition of the node samples is found, even if it requires to
#         effectively inspect more than ``max_features`` features.

#     max_leaf_nodes : int, default=None
#         Grow trees with ``max_leaf_nodes`` in best-first fashion.
#         Best nodes are defined as relative reduction in impurity.
#         If None then unlimited number of leaf nodes.

#     min_impurity_decrease : float, default=0.0
#         A node will be split if this split induces a decrease of the impurity
#         greater than or equal to this value.

#         The weighted impurity decrease equation is the following::

#             N_t / N * (impurity - N_t_R / N_t * right_impurity
#                                 - N_t_L / N_t * left_impurity)

#         where ``N`` is the total number of samples, ``N_t`` is the number of
#         samples at the current node, ``N_t_L`` is the number of samples in the
#         left child, and ``N_t_R`` is the number of samples in the right child.

#         ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
#         if ``sample_weight`` is passed.

#         .. versionadded:: 0.19

#     min_impurity_split : float, default=None
#         Threshold for early stopping in tree growth. A node will split
#         if its impurity is above the threshold, otherwise it is a leaf.

#         .. deprecated:: 0.19
#            ``min_impurity_split`` has been deprecated in favor of
#            ``min_impurity_decrease`` in 0.19. The default value of
#            ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
#            will be removed in 0.25. Use ``min_impurity_decrease`` instead.

#     bootstrap : bool, default=False
#         Whether bootstrap samples are used when building trees. If False, the
#         whole dataset is used to build each tree.

#     oob_score : bool, default=False
#         Whether to use out-of-bag samples to estimate the R^2 on unseen data.

#     n_jobs : int, default=None
#         The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
#         :meth:`decision_path` and :meth:`apply` are all parallelized over the
#         trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
#         context. ``-1`` means using all processors. See :term:`Glossary
#         <n_jobs>` for more details.

#     random_state : int or RandomState, default=None
#         Controls 3 sources of randomness:

#         - the bootstrapping of the samples used when building trees
#           (if ``bootstrap=True``)
#         - the sampling of the features to consider when looking for the best
#           split at each node (if ``max_features < n_features``)
#         - the draw of the splits for each of the `max_features`

#         See :term:`Glossary <random_state>` for details.

#     verbose : int, default=0
#         Controls the verbosity when fitting and predicting.

#     warm_start : bool, default=False
#         When set to ``True``, reuse the solution of the previous call to fit
#         and add more estimators to the ensemble, otherwise, just fit a whole
#         new forest. See :term:`the Glossary <warm_start>`.

#     ccp_alpha : non-negative float, default=0.0
#         Complexity parameter used for Minimal Cost-Complexity Pruning. The
#         subtree with the largest cost complexity that is smaller than
#         ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
#         :ref:`minimal_cost_complexity_pruning` for details.

#         .. versionadded:: 0.22

#     max_samples : int or float, default=None
#         If bootstrap is True, the number of samples to draw from X
#         to train each base estimator.

#         - If None (default), then draw `X.shape[0]` samples.
#         - If int, then draw `max_samples` samples.
#         - If float, then draw `max_samples * X.shape[0]` samples. Thus,
#           `max_samples` should be in the interval `(0, 1)`.

#         .. versionadded:: 0.22

#     Attributes
#     ----------
#     base_estimator_ : ExtraTreeRegressor
#         The child estimator template used to create the collection of fitted
#         sub-estimators.

#     estimators_ : list of DecisionTreeRegressor
#         The collection of fitted sub-estimators.

#     feature_importances_ : ndarray of shape (n_features,)
#         The impurity-based feature importances.
#         The higher, the more important the feature.
#         The importance of a feature is computed as the (normalized)
#         total reduction of the criterion brought by that feature.  It is also
#         known as the Gini importance.

#         Warning: impurity-based feature importances can be misleading for
#         high cardinality features (many unique values). See
#         :func:`sklearn.inspection.permutation_importance` as an alternative.

#     n_features_ : int
#         The number of features.

#     n_outputs_ : int
#         The number of outputs.

#     oob_score_ : float
#         Score of the training dataset obtained using an out-of-bag estimate.
#         This attribute exists only when ``oob_score`` is True.

#     oob_prediction_ : ndarray of shape (n_samples,)
#         Prediction computed with out-of-bag estimate on the training set.
#         This attribute exists only when ``oob_score`` is True.

#     See Also
#     --------
#     sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
#     RandomForestRegressor: Ensemble regressor using trees with optimal splits.

#     Notes
#     -----
#     The default values for the parameters controlling the size of the trees
#     (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
#     unpruned trees which can potentially be very large on some data sets. To
#     reduce memory consumption, the complexity and size of the trees should be
#     controlled by setting those parameter values.

#     References
#     ----------
#     .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
#            Machine Learning, 63(1), 3-42, 2006.

#     Examples
#     --------
#     >>> from sklearn.datasets import load_diabetes
#     >>> from sklearn.model_selection import train_test_split
#     >>> from sklearn.ensemble import ExtraTreesRegressor
#     >>> X, y = load_diabetes(return_X_y=True)
#     >>> X_train, X_test, y_train, y_test = train_test_split(
#     ...     X, y, random_state=0)
#     >>> reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(
#     ...    X_train, y_train)
#     >>> reg.score(X_test, y_test)
#     0.2708...
#     """
#     @_deprecate_positional_args
#     def __init__(self,
#                  n_estimators=100, *,
#                  criterion="mse",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  bootstrap=False,
#                  oob_score=False,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False,
#                  ccp_alpha=0.0,
#                  max_samples=None):
#         super().__init__(
#             base_estimator=ExtraTreeRegressor(),
#             n_estimators=n_estimators,
#             estimator_params=("criterion", "max_depth", "min_samples_split",
#                               "min_samples_leaf", "min_weight_fraction_leaf",
#                               "max_features", "max_leaf_nodes",
#                               "min_impurity_decrease", "min_impurity_split",
#                               "random_state", "ccp_alpha"),
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start,
#             max_samples=max_samples)

#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_features = max_features
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.min_impurity_split = min_impurity_split
#         self.ccp_alpha = ccp_alpha


# class RandomTreesEmbedding(BaseForest):
#     """
#     An ensemble of totally random trees.

#     An unsupervised transformation of a dataset to a high-dimensional
#     sparse representation. A datapoint is coded according to which leaf of
#     each tree it is sorted into. Using a one-hot encoding of the leaves,
#     this leads to a binary coding with as many ones as there are trees in
#     the forest.

#     The dimensionality of the resulting representation is
#     ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
#     the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

#     Read more in the :ref:`User Guide <random_trees_embedding>`.

#     Parameters
#     ----------
#     n_estimators : int, default=100
#         Number of trees in the forest.

#         .. versionchanged:: 0.22
#            The default value of ``n_estimators`` changed from 10 to 100
#            in 0.22.

#     max_depth : int, default=5
#         The maximum depth of each tree. If None, then nodes are expanded until
#         all leaves are pure or until all leaves contain less than
#         min_samples_split samples.

#     min_samples_split : int or float, default=2
#         The minimum number of samples required to split an internal node:

#         - If int, then consider `min_samples_split` as the minimum number.
#         - If float, then `min_samples_split` is a fraction and
#           `ceil(min_samples_split * n_samples)` is the minimum
#           number of samples for each split.

#         .. versionchanged:: 0.18
#            Added float values for fractions.

#     min_samples_leaf : int or float, default=1
#         The minimum number of samples required to be at a leaf node.
#         A split point at any depth will only be considered if it leaves at
#         least ``min_samples_leaf`` training samples in each of the left and
#         right branches.  This may have the effect of smoothing the model,
#         especially in regression.

#         - If int, then consider `min_samples_leaf` as the minimum number.
#         - If float, then `min_samples_leaf` is a fraction and
#           `ceil(min_samples_leaf * n_samples)` is the minimum
#           number of samples for each node.

#         .. versionchanged:: 0.18
#            Added float values for fractions.

#     min_weight_fraction_leaf : float, default=0.0
#         The minimum weighted fraction of the sum total of weights (of all
#         the input samples) required to be at a leaf node. Samples have
#         equal weight when sample_weight is not provided.

#     max_leaf_nodes : int, default=None
#         Grow trees with ``max_leaf_nodes`` in best-first fashion.
#         Best nodes are defined as relative reduction in impurity.
#         If None then unlimited number of leaf nodes.

#     min_impurity_decrease : float, default=0.0
#         A node will be split if this split induces a decrease of the impurity
#         greater than or equal to this value.

#         The weighted impurity decrease equation is the following::

#             N_t / N * (impurity - N_t_R / N_t * right_impurity
#                                 - N_t_L / N_t * left_impurity)

#         where ``N`` is the total number of samples, ``N_t`` is the number of
#         samples at the current node, ``N_t_L`` is the number of samples in the
#         left child, and ``N_t_R`` is the number of samples in the right child.

#         ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
#         if ``sample_weight`` is passed.

#         .. versionadded:: 0.19

#     min_impurity_split : float, default=None
#         Threshold for early stopping in tree growth. A node will split
#         if its impurity is above the threshold, otherwise it is a leaf.

#         .. deprecated:: 0.19
#            ``min_impurity_split`` has been deprecated in favor of
#            ``min_impurity_decrease`` in 0.19. The default value of
#            ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
#            will be removed in 0.25. Use ``min_impurity_decrease`` instead.

#     sparse_output : bool, default=True
#         Whether or not to return a sparse CSR matrix, as default behavior,
#         or to return a dense array compatible with dense pipeline operators.

#     n_jobs : int, default=None
#         The number of jobs to run in parallel. :meth:`fit`, :meth:`transform`,
#         :meth:`decision_path` and :meth:`apply` are all parallelized over the
#         trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
#         context. ``-1`` means using all processors. See :term:`Glossary
#         <n_jobs>` for more details.

#     random_state : int or RandomState, default=None
#         Controls the generation of the random `y` used to fit the trees
#         and the draw of the splits for each feature at the trees' nodes.
#         See :term:`Glossary <random_state>` for details.

#     verbose : int, default=0
#         Controls the verbosity when fitting and predicting.

#     warm_start : bool, default=False
#         When set to ``True``, reuse the solution of the previous call to fit
#         and add more estimators to the ensemble, otherwise, just fit a whole
#         new forest. See :term:`the Glossary <warm_start>`.

#     Attributes
#     ----------
#     estimators_ : list of DecisionTreeClassifier
#         The collection of fitted sub-estimators.

#     References
#     ----------
#     .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
#            Machine Learning, 63(1), 3-42, 2006.
#     .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
#            visual codebooks using randomized clustering forests"
#            NIPS 2007

#     Examples
#     --------
#     >>> from sklearn.ensemble import RandomTreesEmbedding
#     >>> X = [[0,0], [1,0], [0,1], [-1,0], [0,-1]]
#     >>> random_trees = RandomTreesEmbedding(
#     ...    n_estimators=5, random_state=0, max_depth=1).fit(X)
#     >>> X_sparse_embedding = random_trees.transform(X)
#     >>> X_sparse_embedding.toarray()
#     array([[0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
#            [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
#            [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
#            [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
#            [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.]])
#     """

#     criterion = 'mse'
#     max_features = 1

#     @_deprecate_positional_args
#     def __init__(self,
#                  n_estimators=100, *,
#                  max_depth=5,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  sparse_output=True,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False):
#         super().__init__(
#             base_estimator=ExtraTreeRegressor(),
#             n_estimators=n_estimators,
#             estimator_params=("criterion", "max_depth", "min_samples_split",
#                               "min_samples_leaf", "min_weight_fraction_leaf",
#                               "max_features", "max_leaf_nodes",
#                               "min_impurity_decrease", "min_impurity_split",
#                               "random_state"),
#             bootstrap=False,
#             oob_score=False,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start,
#             max_samples=None)

#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.min_impurity_split = min_impurity_split
#         self.sparse_output = sparse_output

#     def _set_oob_score(self, X, y):
#         raise NotImplementedError("OOB score not supported by tree embedding")

#     def fit(self, X, y=None, sample_weight=None):
#         """
#         Fit estimator.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The input samples. Use ``dtype=np.float32`` for maximum
#             efficiency. Sparse matrices are also supported, use sparse
#             ``csc_matrix`` for maximum efficiency.

#         y : Ignored
#             Not used, present for API consistency by convention.

#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted. Splits
#             that would create child nodes with net zero or negative weight are
#             ignored while searching for a split in each node. In the case of
#             classification, splits are also ignored if they would result in any
#             single class carrying a negative weight in either child node.

#         Returns
#         -------
#         self : object

#         """
#         self.fit_transform(X, y, sample_weight=sample_weight)
#         return self

#     def fit_transform(self, X, y=None, sample_weight=None):
#         """
#         Fit estimator and transform dataset.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             Input data used to build forests. Use ``dtype=np.float32`` for
#             maximum efficiency.

#         y : Ignored
#             Not used, present for API consistency by convention.

#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted. Splits
#             that would create child nodes with net zero or negative weight are
#             ignored while searching for a split in each node. In the case of
#             classification, splits are also ignored if they would result in any
#             single class carrying a negative weight in either child node.

#         Returns
#         -------
#         X_transformed : sparse matrix of shape (n_samples, n_out)
#             Transformed dataset.
#         """
#         X = check_array(X, accept_sparse=['csc'])
#         if issparse(X):
#             # Pre-sort indices to avoid that each individual tree of the
#             # ensemble sorts the indices.
#             X.sort_indices()

#         rnd = check_random_state(self.random_state)
#         y = rnd.uniform(size=X.shape[0])
#         super().fit(X, y, sample_weight=sample_weight)

#         self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
#         return self.one_hot_encoder_.fit_transform(self.apply(X))

#     def transform(self, X):
#         """
#         Transform dataset.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             Input data to be transformed. Use ``dtype=np.float32`` for maximum
#             efficiency. Sparse matrices are also supported, use sparse
#             ``csr_matrix`` for maximum efficiency.

#         Returns
#         -------
#         X_transformed : sparse matrix of shape (n_samples, n_out)
#             Transformed dataset.
#         """
#         check_is_fitted(self)
#         return self.one_hot_encoder_.transform(self.apply(X))
