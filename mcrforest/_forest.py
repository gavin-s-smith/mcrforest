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
import copy
import pandas as pd
from tqdm import tqdm
import io
from contextlib import redirect_stdout
from matplotlib.backends.backend_pdf import PdfPages


__all__ = ["RandomForestClassifier",
           "RandomForestRegressor",
           "ExtraTreesClassifier",
           "ExtraTreesRegressor",
           "RandomTreesEmbedding"]


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
        self.estimators_m = {}
        self.estimators_p = {}
        self.new_tree_equivilents_m = {}
        self.new_tree_equivilents_p = {}
        self.estimators_history = []
        self.human_readable_history = []
        self.mcr_history = []
        self.mcr_cache= {}



    def print_unique_trees(self, col_names, indices2print = None):
        
        unique_trees_as_str = set()

        if is_classifier(self.estimators_[0]):
            print('Probabily table to class mapping: {}'.format( self.estimators_[0].classes_) )
        
        print("Printing unique trees..")
        
        for i,e in enumerate(self.estimators_):
            if not indices2print is None:
                if not i in indices2print:
                    continue

            with io.StringIO() as buf, redirect_stdout(buf):
                
                e.tree_.print_tree(col_names)
                unique_trees_as_str.add(buf.getvalue())
        
        for t in unique_trees_as_str:
            print(t)



    def print_trees(self, col_names, indices2print = None):

        for i,e in enumerate(self.estimators_):
            if not indices2print is None:
                if not i in indices2print:
                    continue

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


    def undo_mcr_step(self, verbose = 0):
        """
        Returns the human readable step that was undone.
        """
        if len(self.estimators_history) == 0:
            print('Nothing to undo.')
            return None
        
        self.estimators_ = self.estimators_history.pop()

        rtn = self.human_readable_history.pop()
        
        if verbose > 0:
            print(f'Undid step: {rtn}')
            print(f'Current history: {"-->".join(self.human_readable_history)}')

        return rtn


    def get_mcr_state(self):

        return f'-->{",".join(self.human_readable_history)}'

    def set_mcr_state(self, force_use, var_indexs, debug = True, verbose = 0 ):
        
        if not isinstance(var_indexs, np.ndarray):
            raise Exception(f'var_indexes is expected to be a numpy array, was {var_indexs} of type(var_indexs): {type(var_indexs)}')
        
        for l in var_indexs:
            self.mcr_history.append( [l, force_use] )

        self.estimators_history.append( self.estimators_.copy() )
        if force_use:
            self.human_readable_history.append( f'USE({var_indexs})' )
        else:
           self.human_readable_history.append( f'AVOID({var_indexs})' ) 

        if debug:
            print(f' set_estimators(self, force_use = {force_use}, var_indexs = {var_indexs}, debug = True )')

        if force_use:
            self.estimators_ = self.estimators_p[str(var_indexs)].copy()
            self.forest_equivilents = self.new_tree_equivilents_p[str(var_indexs)].copy()
        else:
            self.estimators_ = self.estimators_m[str(var_indexs)].copy()
            self.forest_equivilents = self.new_tree_equivilents_m[str(var_indexs)].copy()
        
        if verbose > 0:
            print(f'Did step: {self.human_readable_history[-1]}')
            print(f'Current history: {"-->".join(self.human_readable_history)}')


    def get_specific_forest_from_mcr_set(self, var_idx, force_use = True):
        rfcopy = copy.deepcopy(self)

        # Fix the tree swap
        rfcopy.set_mcr_state(force_use, np.asarray([var_idx]))

        # Now sort the surrogates
        for e in rfcopy.estimators_:
            # for each node in each tree we will "fix" it according the MCR rules
            # this is inplace and currently has no undo mechanism. Hence must be done on a copy
            e.mcr_freeze( var_idx, force_use )

        
        return rfcopy


    def predict_tree(self, tree_estimator, X ):
        rtn = tree_estimator.predict(X)
            
        if is_classifier(self):
            # By default a sklearn RF will first convert the input classes to 0,1,2,3... etc.
            # These will then be what the trees are trained off. I.e. y is mapped at the forest level (inputs 1,2) and trees learn and predict (0,1)
            # Since here we are at the forest level we must map it back
            rtn = self.classes_.take(rtn.astype(np.int64),axis=0)    

        return rtn
    
    def predict_vim_tree(self, tree_estimator, X_perm, indices_to_permute, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post):
        rtn = tree_estimator.predict_vim(X_perm, indices_to_permute, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)
            
        if is_classifier(self):
            # By default a sklearn RF will first convert the input classes to 0,1,2,3... etc.
            # These will then be what the trees are trained off. I.e. y is mapped at the forest level (inputs 1,2) and trees learn and predict (0,1)
            # Since here we are at the forest level we must map it back
            rtn = self.classes_.take(rtn.astype(np.int64),axis=0)    

        return rtn


    

    def mcr_shap_plot(self, X, mcr_plus = True, plot_size = None, sort = True):
        print('WARNING: This function is still in development. You must have a patched version of SHAP for this to work.')
        
        # Check if 
        # (1) the model has been fit
        # (2) MCR has been called with no groupings
        print('WARNING: Ensure that the model has been fit')
        print('WARNING: Ensure plot_mcr(...) has been called with no groupings. This is currently not checked.')
        print('UPDATED2')

        import shap
        from sklearn.ensemble import RandomForestClassifier as sklrf
        import seaborn as sns
        if is_classifier(self):
            from sklearn.tree import DecisionTreeClassifier as skltree
        else:
            from sklearn.tree import DecisionTreeRegressor as skltree

        cmp_min = sns.dark_palette("#FF8A57FF", reverse=False, as_cmap=True)
        cmp_max = sns.dark_palette("#97BC62FF", reverse=False, as_cmap=True)

        ### decided to do it as a dictionary
        ### as opposed to a list as easier to manipulate later


        rtn_mcr_plus = []

        for i, var in enumerate(X.columns.tolist()):
            mm = self.get_specific_forest_from_mcr_set(var_idx = X.columns.tolist().index(var), force_use = mcr_plus)
            
            old_class = mm.__class__
            old_tree_class = mm.estimators_[0].__class__
            
            mm.__class__ = sklrf
            for i,e in enumerate(mm.estimators_):
                mm.estimators_[i].__class__ = lambda: sklearn.tree._tree.Tree  
            explainerm = shap.TreeExplainer(mm, X, check_additivity=False)
            shap_values_randomm = explainerm.shap_values(X, check_additivity=False)
            rtn_mcr_plus.append( shap_values_randomm[1][:,i] ) 

            mm.__class__ = old_class
            for i,e in enumerate(mm.estimators_):
                mm.estimators_[i].__class__= old_tree_class

        
        if mcr_plus:
            cmp = cmp_max
        else:
            cmp = cmp_min

        shap.summary_plot(np.asarray(rtn_mcr_plus).T, X, show = False, sort = sort, plot_size = plot_size, cmap=cmp, max_display = X.shape[1])

        # returns the SHAP values (# samples x # features).
        return np.asarray(rtn_mcr_plus).T


    def mcr(self, X_in, y_in, indices_to_permute, 
                                    num_times = 100, debug = False, debug_call = False, debug_trees = None,
                                    mcr_type = 1, restrict_trees_to = None, mcr_as_ratio = False, seed = 13111985, 
                                    enable_Tplus_transform = True, custom_scorer = None
                                    ):
        """ Computes MCR+ or MCR-

        Parameters
        ----------
        X_in: numpy.array
                The input feature matrix
        y_in: numpy.array
                The output feature array
        indices_to_permute: np.array
                The indicies to permute. Typically a one element array denoting the varaible of interest for MCR.
        num_times: int
                Number of times variable will be permuted when computing the MDA for MCR.
        debug: bool
            If True the method returns: 
            1. Mean performance of the new forest using both surrogates and tree replacement
            2. Mean performance of the new forest using only surrogates
            3. Mean performance of the original forest
            4. A string with the min, mean and max amount the trees left differed from their reference tree (i.e. amount left the 0-Rashomon set)
        debug_call: bool
            If True print the function call showing a select set (hardcoded) parameters values to aid in debuging along with the value returned (if debug = False).
        debug_trees: None or Array
            If not None, then the array must be the column names for X_in. Trees will be printed based on this information along with other debug information for each tree.
        mcr_type: int [-1 or 1]
            For MCR+: 1, for MCR-: -1
        restrict_trees_to: int
                If not None, only use the first n trees in the reference forest. 
        mcr_as_ratio: bool
                    Compute the MCR ratio (as per Fisher et. al's 2019 JMLR paper) rather than the difference in predictive performance. 
                    Not tested outside of the specific dataset used within that paper and Smith et. al. Model Class Reliance for Random Forests (2020). 
                    SHOULD NOT BE USED (i.e. mcr_as_ratio should remain set to False )
        seed: int
            Random number generator seed.
        enable_Tplus_transform: bool
                            If True the tree transform will be used. 
                            NOTE: If this parameter is set to False then this function no longer computes the MCR. 
                                  See G. SMITH, R. MANSILLA and J. GOULDING, 2020. Model Class Reliance for Random Forests. In 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada
        custom_scorer: None or a custom function with signiture: score_func(y, y_pred) where y and y_pred are arrays and the function returns a single score.
        """


        
        is_classification = is_classifier(self)

        if not isinstance(y_in, np.ndarray):
            raise Exception('y_in must be a numpy array')
        if not len(y_in.shape) == 1:
            raise Exception('y_in must be a 1D numpy array')

        if mcr_type == 1:
            mcr_plus = True
        elif mcr_type == -1:
            mcr_plus = False
        else:
            raise Exception(f'mcr_type must be either 1 or -1, was: {mcr_type}')
        
        #mcr_ordering: numpy.array
        #            a 1D numpy array of input variable indices indicating which variables must be used before others (Left to Right in the array)

        #mcr_ordering = self.compute_mcr_order_list( X_in.shape[1], indices_to_permute, mcr_plus )

        mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post = self.compute_mcr_order_list_pre_others_post( X_in.shape[1], indices_to_permute, mcr_plus )

        #print(f'mcr_ordering: {mcr_ordering}')

        if debug_call:
            print(f'mcr(self, X_in, y_in, indices_to_permute = {indices_to_permute}, num_times = {num_times}, mcr_type = {mcr_type}, mcr_ordering_pre = {mcr_ordering_pre}, mcr_ordering_others = {mcr_ordering_others}, mcr_ordering_post = {mcr_ordering_post}, seed = {seed}, ...')


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
        y = np.tile(y_in,num_times)


        
        n_samples = len(y)

        # permutation is done by duplicating X by n_times and computing the Decrease in Accuracy for each instance independtly.
        # Given the number samples in each permutation is the same, 
        # the MDA doing it this way vs. grouping -> taking the MDA -> taking the Mean of the groups is the same. 
        # NOTE: This is slightly different to V1, as the truffle shuffle (tree shuffle) is done only once based on the overall MDA
        #       not for each permtation. This change is required to realize a specific instance of a forest at the end.

        #for i_num_times in range(num_times):
        # Make a copy of X and permute
        X_perm = np.tile(X_in, (num_times,1) )
        X = np.tile(X_in, (num_times,1) )
    
        for i_permidx in indices_to_permute:
            np.random.shuffle(X_perm[:, i_permidx])

        
                    

        # for each tree we will predict all samples and store them here
        per_fplus_tree_preds = np.ones([n_trees, n_samples])*-9999
        per_ref_tree_preds = np.ones([n_trees, n_samples])*-9999
        
        # for each tree make predictions for all samples using f+




        def collate_parallel( eidx ):
            # x [0] is the id
            per_ref_tree_preds[eidx,:] = self.predict_tree(self.estimators_[eidx],X)
    
            per_fplus_tree_preds[eidx,:] = self.predict_vim_tree(self.estimators_[eidx],X_perm, indices_to_permute, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)

        if self.n_jobs is None or self.n_jobs == 1:

            for eidx in range(n_trees):
                
                

                per_ref_tree_preds[eidx,:] = self.predict_tree(self.estimators_[eidx],X)
                
                per_fplus_tree_preds[eidx,:] = self.predict_vim_tree(self.estimators_[eidx],X_perm, indices_to_permute, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)
        
        else:
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose, **_joblib_parallel_args(require="sharedmem"))(delayed(collate_parallel)(eidx) for eidx in range(n_trees))
                
                                                                   
        
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
        
        if not debug_trees is None:
            for eidx in range(n_trees):
                #if per_tree_diff_in_loss_ref_tree_vs_fplus_tree[eidx] != 0:
                #    continue
                print(f'\nTree index: {eidx}. Loss: {per_tree_diff_in_loss_ref_tree_vs_fplus_tree[eidx]}')
                self.estimators_[eidx].print_tree(debug_trees)
                
        


        # Build up a new forest (the one we will take the MR of to get MCR+/-)
        # We do this by storing trees by index (into the trees used by the reference model)

        #############
        # BEING BUILD NEW FOREST

        new_trees_indexes = []
        new_tree_equivilents = []
        for i_ntrees in range(n_trees):
            if enable_Tplus_transform:
                # for this tree get the set of trees that have accuacy equiviliency
                indexs_of_eq_forests = self.forest_equivilents[i_ntrees]
                
                
                if mcr_type > 0:
                    # If we have MCR+ we want to select the tree with the largest change in predictive peformance (most damage)
                    min_or_max = np.max
                else:
                    # If we have MCR+ we want to select the tree with the smallest change in predictive peformance (least damage)
                    min_or_max = np.min
                
                # Only consider those trees that are the min (max) [MCR- (MCR+)] with no tolerance. 
                # Could also have arg-sorted the threshold set and selected the first element.
                equally_best_worst_tree_set = np.asarray(indexs_of_eq_forests)[ 
                                                        np.abs(per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] - 
                                                        min_or_max(per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ])) 
                                                        <= 0
                                                ]
                # The threhold set. A superset of equally_best_worst_tree_set. Should really only compute one.
                equally_best_worst_tree_set_theshold = np.asarray(indexs_of_eq_forests)[ 
                                                        np.abs(per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] - 
                                                        min_or_max(per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ])) 
                                                        <= 0.001 #self.mcr_tree_equivilient_tol
                                                ]
                # Select the equally performant tree that will be used in the new forest instead of the current one (indexed by i_ntrees)
                # which is most (least) [MCR+ (MCR-)] relient on the varaible (or variable group) or interst.
                if i_ntrees in equally_best_worst_tree_set:
                    new_trees_indexes.append( i_ntrees ) 
                else:
                    select_idx = np.random.randint(0, high=len(equally_best_worst_tree_set), size=None, dtype=int)
                    new_trees_indexes.append( equally_best_worst_tree_set[select_idx] ) # take a random one

                # set the list of trees that (1) has equal predictive performance and (2) relies on the varaible (set) of interest to the
                # same degree (where same is defined byt the above threshold)
                new_tree_equivilents.append(equally_best_worst_tree_set_theshold)

                #print(f'TreeID: {i_ntrees} mcr_type: {mcr_type} equally_best_worst_tree_set: {equally_best_worst_tree_set} per_tree_diff_in_loss_ref_tree_vs_fplus_tree: {per_tree_diff_in_loss_ref_tree_vs_fplus_tree}')
            else:
                new_trees_indexes.append(i_ntrees)

                new_tree_equivilents.append(indexs_of_eq_forests)
        
        #print(new_trees_indexes)
        #print(new_tree_equivilents)
        
        if custom_scorer is None:
                
            if is_classification:
                forest_scorer = lambda x: np.mean(self.predict_vim(x,indices_to_permute,mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)==y)
            else:
                if self.get_params()['criterion'] == 'mse':
                    forest_scorer = lambda x: np.mean( (self.predict_vim(x,indices_to_permute,mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)-y)**2 )
                else:
                    forest_scorer = lambda x: np.mean( np.abs(self.predict_vim(x,indices_to_permute,mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)-y) )

            if is_classification:
                forest_scorer_reference_model = lambda x: np.mean(self.predict(x)==y)
            else:
                if self.get_params()['criterion'] == 'mse':
                    forest_scorer_reference_model = lambda x: np.mean( (self.predict(x)-y)**2 )
                else:
                    forest_scorer_reference_model = lambda x: np.mean( np.abs(self.predict(x)-y) )
        else:

            def cust_fn_vim( X ):
                y_pred = self.predict_vim(X,indices_to_permute,mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post)
                return custom_scorer(y,y_pred)
            
            def cust_fn( X ):
                y_pred = self.predict(X)
                return custom_scorer(y,y_pred)

            forest_scorer = cust_fn_vim
            forest_scorer_reference_model = cust_fn

        ref_forest_orig_data_score = forest_scorer(X)

        #acc_no_per_surrogates_VK = forest_scorer(X)
        if debug:
            acc_with_per_surrogates_VK = forest_scorer(X_perm)

        ##############################################
        # CHANGE OUR FOREST TO THE NEW FOREST
        ##############################################
        self.estimators_ = np.asarray(real_estimators_)[new_trees_indexes]
        # if indices_to_permute[0] == 4:
        #     print(new_trees_indexes)
        #     print(per_tree_diff_in_loss_ref_tree_vs_fplus_tree[ indexs_of_eq_forests ] )
        #     print('p')
        
        # store the 
        # if len(indices_to_permute) == 1:
        #     if mcr_type < 0:
        #         self.estimators_m[indices_to_permute[0]] = np.asarray(real_estimators_)[new_trees_indexes]
        #         self.new_tree_equivilents_m[indices_to_permute[0]] = np.copy(new_tree_equivilents)
        #     else:
        #         self.estimators_p[indices_to_permute[0]] = np.asarray(real_estimators_)[new_trees_indexes]
        #         self.new_tree_equivilents_p[indices_to_permute[0]] = np.copy(new_tree_equivilents)
        # else:
        #     print('WANRING: CURRENTLY FOR v2: len(indices_to_permute) > 1 IS EXPERIMENTAL ')
        if mcr_type < 0:
            self.estimators_m[str(indices_to_permute)] = np.asarray(real_estimators_)[new_trees_indexes]
            self.new_tree_equivilents_m[str(indices_to_permute)] = [np.copy(x) for x in new_tree_equivilents]
        else:
            #print(f'Setting: self.estimators_p[str(indices_to_permute)]: {str(indices_to_permute)}')
            self.estimators_p[str(indices_to_permute)] = np.asarray(real_estimators_)[new_trees_indexes]
            # print(f'est ----> {new_trees_indexes}')
            # print(f'type ----> {type(new_tree_equivilents)}')
            # print(f'----> {new_tree_equivilents}')
            self.new_tree_equivilents_p[str(indices_to_permute)] = [ np.copy(x) for x in new_tree_equivilents ]

        # Check if we are still in the Rashomon set
        new_forest_orig_data_score = forest_scorer(X)
        
        rashomon_set_error = np.abs(ref_forest_orig_data_score - new_forest_orig_data_score)
        amount_left_set_by.append(rashomon_set_error)
        #if  rashomon_set_error > 0.00001:
        #    print('WARNING: Left the Rashomon set. Original acc/squared error: {}, After: {}, Difference: {}'.format(ref_forest_orig_data_score,new_forest_orig_data_score,rashomon_set_error))

        
        #new_forest_orig_data_score = forest_scorer(X)
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

            
   


        if debug:

            return np.mean(acc_set_sur_and_truffle), np.mean(acc_set_sur_only),np.mean(acc_set_ref_model), 'min: {:.5f}, mean: {:.5f}, max: {:.5f}'.format(np.min(amount_left_set_by), np.mean(amount_left_set_by), np.max(amount_left_set_by)) #new

        if debug_call:
            print(f'Return: {np.mean(acc_set_sur_and_truffle)}')

        return np.mean(acc_set_sur_and_truffle)

    def compute_mcr_order_list(self, total_num_features, indexes_of_interest, mcr_plus):
        
        must_use_variable_ordering = []
        
        # all variable not in history or the current variable
        all_other_vars_as_list = []
        for jidx in range(total_num_features):
            if jidx not in indexes_of_interest and jidx not in [item[0] for item in self.mcr_history]: 
                all_other_vars_as_list.append( jidx )

        # For MCR-/+ Build must_use_variable_ordering for this variable and the history
        for e in self.mcr_history:
            if e[1] == True: # must use
                must_use_variable_ordering.append(e[0])

        if mcr_plus:
        
            if all([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]): # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                for index in indexes_of_interest:
                    must_use_variable_ordering.append(index)
            elif any([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]):
                raise Exception('Grouped variables cannot partially appear in the history, but somehow have.')

            must_use_variable_ordering += all_other_vars_as_list
        else:
            must_use_variable_ordering += all_other_vars_as_list

            if all([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]): # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                for index in indexes_of_interest:
                    must_use_variable_ordering.append(index)
            elif any([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]):
                raise Exception('Grouped variables cannot partially appear in the history, but somehow have.')


        for e in self.mcr_history[::-1]:
            if e[1] == False: 
                must_use_variable_ordering.append(e[0]) # append variable you want to avoid using

        # convert must_use_variable_ordering into a 1D numpy array to run mcr_ordering
        must_use_variable_ordering = np.array(must_use_variable_ordering)
        
        return must_use_variable_ordering
        

    def compute_mcr_order_list_pre_others_post(self, total_num_features, indexes_of_interest, mcr_plus):    
        
        must_use_variable_ordering = []

        pre = []
        others = []
        post = []
        
        # all variable not in history or the current variable
        all_other_vars_as_list = []
        for jidx in range(total_num_features):
            if jidx not in indexes_of_interest and jidx not in [item[0] for item in self.mcr_history]: 
                all_other_vars_as_list.append( jidx )

        # For MCR-/+ Build must_use_variable_ordering for this variable and the history
        for e in self.mcr_history:
            if e[1] == True: # must use
                must_use_variable_ordering.append(e[0])

        if mcr_plus:
        
            if all([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]): # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                for index in indexes_of_interest:
                    must_use_variable_ordering.append(index)
            elif any([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]):
                raise Exception('Grouped variables cannot partially appear in the history, but somehow have.')

            pre = must_use_variable_ordering
            others = all_other_vars_as_list

            #must_use_variable_ordering += all_other_vars_as_list
        else:
            pre = must_use_variable_ordering
            others = all_other_vars_as_list
            #must_use_variable_ordering += all_other_vars_as_list
            post = []

            if all([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]): # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                for index in indexes_of_interest:
                    post.append(index)
            elif any([x not in [x[0] for x in self.mcr_history] for x in indexes_of_interest]):
                raise Exception('Grouped variables cannot partially appear in the history, but somehow have.')


        for e in self.mcr_history[::-1]:
            if e[1] == False: 
                post.append(e[0]) # append variable you want to avoid using

        # convert must_use_variable_ordering into a 1D numpy array to run mcr_ordering
        #must_use_variable_ordering = np.array(must_use_variable_ordering)
        
        return np.asarray(pre, dtype = np.int64), np.asarray(others, dtype = np.int64), np.asarray(post, dtype = np.int64)

    def plot_mcr(self,X_in, y_in, feature_names = None, feature_groups_of_interest = 'all individual features', num_times = 100, show_fig = True, pdf_file = None, use_cache = False, include_permutation_importance = None, custom_scorer = None):
            
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
        pdf_file: str
                If not None, a path to save a pdf of the graph to.
        use_cache: bool
                If True save the resulting MCR frame to the variable self.mcr_cache using the USE/AVOID human readable history string.
        include_permutation_importance: bool
                If True compute and plot the unconditional permutation importance scores. 
        custom_scorer: None of callable
                If not None, a Loss function (or loss function) with signature score_func(y, y_pred). Assumes smaller is better.
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

        
        if isinstance(y_in, pd.DataFrame) or isinstance(y_in, pd.core.series.Series): 
            y = y_in.values
        else:
            y = y_in

        
        results = []
        if isinstance(feature_groups_of_interest, str):
            if feature_groups_of_interest == 'all individual features':
                groups_of_indicies_to_permute = [[x] for x in range(len(feature_names))]
            else:
                raise Exception('feature_groups_of_interest incorrectly specified. If not specifying to use all individual features via "all individual features" you must pass a numpy array of numpy arrays. See the documentation on github.')
        elif len(feature_groups_of_interest) != len(feature_names):
            raise Exception(f'The wrong number of feature names were provided. len(feature_groups_of_interest): {len(feature_groups_of_interest)} != len(feature_names) {len(feature_names)}')
        
        elif isinstance(feature_groups_of_interest[0], str) or isinstance(feature_groups_of_interest[0][0], str):
            # we need to convert the feature groups to index groups
            if not isinstance(X_in, pd.DataFrame):
                raise Exception('You can only pass variable names for grouping if you pass the data as a dataframe. X was not a dataframe.')
            X_cols = X_in.columns.tolist()
            fgi = []

            if isinstance(feature_groups_of_interest[0], str):
                for iat in feature_groups_of_interest:
                    fgi.append(X_cols.index(iat))
            else:
                for iat in feature_groups_of_interest:
                    tmp = []
                    for jat in iat:
                        tmp.append(X_cols.index(jat))
                    fgi.append(tmp)
            
            groups_of_indicies_to_permute = fgi
        else:
            groups_of_indicies_to_permute = feature_groups_of_interest
        
        # New MCR+ perm imp
        print('Processing MCR+ groups of features.')
        gp_idx = 0
        for gp in tqdm(groups_of_indicies_to_permute):
            rn = self.mcr(X,y, np.asarray(gp) ,  num_times = num_times, mcr_type = 1, custom_scorer = custom_scorer)
            results.append([feature_names[gp_idx], 'RF-MCR+', rn])
            gp_idx += 1

        # New MCR- perm imp
        print('Processing MCR- groups of features.')
        gp_idx = 0
        for gp in tqdm(groups_of_indicies_to_permute):
            rn = self.mcr(X,y, np.asarray(gp) ,  num_times = num_times,  mcr_type = -1, custom_scorer = custom_scorer)
            results.append([feature_names[gp_idx], 'RF-MCR-', rn])
            gp_idx += 1

        lbl = [ x[0] for x in results if 'MCR+' in x[1] ]
        mcrp = [ x[2] for x in results if 'MCR+' in x[1] ]
        mcrm = [ x[2] for x in results if 'MCR-' in x[1] ]

        print('MCR+ sum: {}'.format(sum(mcrp)))

        rf_results2 = pd.DataFrame({'variable':lbl, 'MCR-':mcrm, 'MCR+':mcrp })

        import seaborn as sns
        import matplotlib.pyplot as plt

        def plot_mcr_graph(df_in, fig_size = (11.7, 8.27)):
            df_in = df_in.copy()
            df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns]
            ax = sns.barplot(x='MCR- (lollypops) | MCR+ (bars)',y='variable',data=df_in)
            plt.gcf().set_size_inches(fig_size)
            plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='skyblue')
            plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'skyblue')
            if 'perm_scores' in df_in.columns.tolist():
                plt.plot(df_in['perm_scores'], range(df_in.shape[0]), "x", color = 'black')


        
        if include_permutation_importance:
            perm_scores = self.unconditional_permutation_importance(X_in, y_in, feature_groups_of_interest = feature_groups_of_interest, feature_names = feature_names, num_times = num_times)
            rf_results2['perm_scores'] = perm_scores
        
        plot_mcr_graph(rf_results2)
        if show_fig:
            plt.show()
        
        if not pdf_file is None:
            with PdfPages(pdf_file) as pdf:
                pdf.savefig()

        if use_cache:
            self.mcr_cache[f'-->{",".join(self.human_readable_history)}'] = rf_results2

        return rf_results2



    # GAVIN TODO: Integrate better
    def unconditional_permutation_importance(self, X_in, y_in, feature_groups_of_interest, feature_names = None, pre_permutated = False, num_times = 100, debug = False, random_state = 13111985):
        
        is_classification = is_classifier(self)
        
        if isinstance(X_in, pd.DataFrame):
            X = X_in.values
            if feature_names is None:
                feature_names = X_in.columns.tolist()
        else:
            X = X_in
            if feature_names is None:
                feature_names = ['f_{}'.format(i) for i in range(X.shape[1])]

        
        if isinstance(y_in, pd.DataFrame) or isinstance(y_in, pd.core.series.Series): 
            y = y_in.values
        else:
            y = y_in

        
        results = []
        if isinstance(feature_groups_of_interest, str):
            if feature_groups_of_interest == 'all individual features':
                groups_of_indicies_to_permute = [[x] for x in range(len(feature_names))]
            else:
                raise Exception('feature_groups_of_interest incorrectly specified. If not specifying to use all individual features via "all individual features" you must pass a numpy array of numpy arrays. See the documentation on github.')
        elif len(feature_groups_of_interest) != len(feature_names):
            raise Exception(f'The wrong number of feature names were provided. len(feature_groups_of_interest): {len(feature_groups_of_interest)} != len(feature_names) {len(feature_names)}')
        
        elif isinstance(feature_groups_of_interest[0], str) or isinstance(feature_groups_of_interest[0][0], str):
            # we need to convert the feature groups to index groups
            if not isinstance(X_in, pd.DataFrame):
                raise Exception('You can only pass variable names for grouping if you pass the data as a dataframe. X was not a dataframe.')
            X_cols = X_in.columns.tolist()
            fgi = []

            if isinstance(feature_groups_of_interest[0], str):
                for iat in feature_groups_of_interest:
                    fgi.append(X_cols.index(iat))
            else:
                for iat in feature_groups_of_interest:
                    tmp = []
                    for jat in iat:
                        tmp.append(X_cols.index(jat))
                    fgi.append(tmp)
            
            groups_of_indicies_to_permute = fgi
        else:
            groups_of_indicies_to_permute = feature_groups_of_interest


        """ Return the accuracy of the prediction of X compared to y. """
        np.random.seed(random_state)
        if is_classification:
            base_score = self.score(X_in,y_in)

        else:
            if self.get_params()['criterion'] == 'mse':
                base_score = mean_squared_error(y_in, self.predict(X_in))
            elif self.get_params()['criterion'] == 'mae':
                base_score = mean_absolute_error(y_in, self.predict(X_in))
            else:
                raise Exception('Unsupported criterion: {}'.format(self.get_params()['criterion']))

        
        rtn_list = []
        for indices_to_permute in tqdm(groups_of_indicies_to_permute):
            r_set = []
        # print('====d=========================')
            for i in range(num_times):
                #print('==========ud==================: {}'.format(indices_to_permute))
                X = X_in.values.copy()
                y = y_in.copy()
                if not pre_permutated:
                    for i in indices_to_permute:

                        np.random.shuffle(X[:, i])

                if is_classification:
                    # MDA
                    r_set.append(base_score- self.score(X,y))
                    
                else:
                    # Here we return Mean increase in Error (MIE)
                    if self.get_params()['criterion'] == 'mse':
                        r_set.append(mean_squared_error(y, self.predict(X)) - base_score)
                    elif self.get_params()['criterion'] == 'mae':
                        r_set.append(mean_absolute_error(y, self.predict(X)) - base_score)
                    else:
                        raise Exception('Unsupported criterion: {}'.format(self.get_params()['criterion']))
                    
            
            mean_performance = np.mean(r_set) #calculate the average accuracy
            rtn_list.append(mean_performance)
        return rtn_list



 





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
            
            for i in range( X.shape[1] ): # check each variable
                
                # if tidx == 482 and i == 0:
                #     print(t.print_tree(col_names = ['b','t','e','h']))

                mcr_ordering_pre = np.asarray([i], dtype = np.int64)
                mcr_ordering_others = np.asarray([ x for x in range(X.shape[1]) if x != i], dtype = np.int64)
                mcr_ordering_post = np.asarray([], dtype = np.int64)

                if is_classifier(self):
                    b = accuracy_score(y, t.predict_vim(X,np.asarray([i], dtype=np.int64), mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post))
                else:
                    b = mean_squared_error(y, t.predict_vim(X,np.asarray([i], dtype=np.int64), mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post))
                
                #b = mean_squared_error(y[5], t.predict_vim(X[5,:].reshape(1,-1),np.asarray([i]), 1))
                #c = mean_squared_error(y, t.predict_vim(X,np.asarray([i], dtype=np.int64), -1))
                if a != b:
                    print('np.asarray([i], dtype=np.int64): {}'.format(np.asarray([i], dtype=np.int64)))
                    print('Bootstrap: {}'.format(self.bootstrap))
                    print('a: {}'.format(a))
                    print('b: {}'.format(b))
                    print(f'Tree idx: {tidx}, variable idx: {i}')
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


    def predict_vim(self, X, permuated_vars, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post ):
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
        
        proba = self.predict_proba_vim(X,permuated_vars, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post )

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

    def predict_proba_vim(self, X,permuted_vars, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post ):
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
            delayed(_accumulate_prediction)(e.predict_proba_vim, (X,permuted_vars, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post ), all_proba,
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


  

    def predict_vim(self, X, permuted_indices, mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post):
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
            delayed(_accumulate_prediction)(e.predict_vim_from_parallel_fn, (X,permuted_indices,mcr_ordering_pre, mcr_ordering_others, mcr_ordering_post), [y_hat], lock)
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
            base_estimator=DecisionTreeClassifier(spoof_as_sklearn=spoof_as_sklearn), # SPLITTER GOES HERE GAVIN 2021, NEED TO UPDATE FOR REGRESSOR TOO
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
        
        if not max_leaf_nodes is None:
            raise Exception('For MCR, max_leaf_nodes is currently not supported.')

        if spoof_as_sklearn:
            from sklearn.ensemble import RandomForestClassifier as sklrfc
            self.__class__ = sklrfc


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

        if not max_leaf_nodes is None:
            raise Exception('For MCR, max_leaf_nodes is currently not supported.')

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
            from sklearn.ensemble import RandomForestRegressor as sklrfr
            self.__class__ = sklrfr


class ExtraTreesClassifier(ForestClassifier):
    """
    An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

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
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

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
    base_estimator_ : ExtraTreesClassifier
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

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    See Also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    ExtraTreesClassifier(random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
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
                 max_samples=None):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
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
            max_samples=max_samples)

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


class ExtraTreesRegressor(ForestRegressor):
    """
    An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

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

    max_features : {"auto", "sqrt", "log2"} int or float, default="auto"
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
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

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
    base_estimator_ : ExtraTreeRegressor
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
        The number of features.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    See Also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import ExtraTreesRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(
    ...    X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.2708...
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
                 max_samples=None):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
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
            max_samples=max_samples)

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


class RandomTreesEmbedding(BaseForest):
    """
    An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    max_depth : int, default=5
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
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
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

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

    sparse_output : bool, default=True
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`transform`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls the generation of the random `y` used to fit the trees
        and the draw of the splits for each feature at the trees' nodes.
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    Examples
    --------
    >>> from sklearn.ensemble import RandomTreesEmbedding
    >>> X = [[0,0], [1,0], [0,1], [-1,0], [0,-1]]
    >>> random_trees = RandomTreesEmbedding(
    ...    n_estimators=5, random_state=0, max_depth=1).fit(X)
    >>> X_sparse_embedding = random_trees.transform(X)
    >>> X_sparse_embedding.toarray()
    array([[0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.]])
    """

    criterion = 'mse'
    max_features = 1

    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 sparse_output=True,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=None)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

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
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Fit estimator and transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        X = check_array(X, accept_sparse=['csc'])
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])
        super().fit(X, y, sample_weight=sample_weight)

        self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
        return self.one_hot_encoder_.fit_transform(self.apply(X))

    def transform(self, X):
        """
        Transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        check_is_fitted(self)
        return self.one_hot_encoder_.transform(self.apply(X))
