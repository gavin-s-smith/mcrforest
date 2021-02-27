# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#cython: language_level=3

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX


import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport PriorityHeap
from ._utils cimport PriorityHeapRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples','surrogate_flip','surrogate_threshold', 'surrogate_feature', 'num_surrogates'  ],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64,(np.intp,(100)),(np.float64,(100)),(np.intp,(100)),np.intp],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).surrogate_flip,
        <Py_ssize_t> &(<Node*> NULL).surrogate_threshold,
        <Py_ssize_t> &(<Node*> NULL).surrogate_feature,
        <Py_ssize_t> &(<Node*> NULL).num_surrogates
    ]
})

# NODE_DTYPE = np.dtype([ ('left_child', np.intp, 1),
#                         ('right_child', np.intp, 1),
#                         ('feature', np.intp, 1),
#                         ('threshold', np.float64, 1),
#                         ('impurity', np.float64, 1),
#                         ('n_node_samples', np.intp, 1),
#                         ('weighted_n_node_samples', np.float64, 1),
#                         ('surrogate_flip', np.intp, 100),
#                         ('surrogate_threshold', np.float64, 100),
#                         ('surrogate_feature', np.intp, 100),
#                         ('num_surrogates', np.intp, 1)
#                     ])

# SIZE_t left_child                    # id of the left child of the node
#     SIZE_t right_child                   # id of the right child of the node
#     SIZE_t feature                       # Feature used for splitting the node
#     DOUBLE_t threshold                   # Threshold value at the node
#     DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
#     SIZE_t n_node_samples                # Number of samples at the node
#     DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node
#     SIZE_t surrogate_flip[100]  # For surrogates it indicates if the condition needs to be flipped. (-1 if it does, 1 if not)
#     DTYPE_t surrogate_threshold[100]  # 
#     SIZE_t surrogate_feature[100]  # 
#     INT32_t num_surrogates

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        #print('Starting Build - Correct for Gavin Modification')
        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        split.num_surrogates = 0

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples,
                                         split.surrogate_flip,
                                            split.surrogate_threshold,
                                            split.surrogate_feature, 
                                            split.num_surrogates)
                
                #with gil:
                #    print(f'---->{tree.nodes[node_id].num_surrogates}')

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()





# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
       
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
    

        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))

        

        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cpdef void mcr_freeze(self, int var_idx, bint force_use ):
        
        cdef SIZE_t temp_node
        
        for i in range(self.node_count-1):
            
            
            if self.nodes[i].num_surrogates <= 0:
                continue # either not a real node or we do not have to do anything
            
            if force_use:
                # MCR+
                if self.nodes[i].feature == var_idx:
                    continue
                else:
                    #print(f'self.nodes[i]: {self.nodes[i]}')
                    for s in range(self.nodes[i].num_surrogates):
                        
                        if self.nodes[i].surrogate_feature[s] == var_idx:
                            self.nodes[i].feature = self.nodes[i].surrogate_feature[s]
                            self.nodes[i].threshold = self.nodes[i].surrogate_threshold[s]
                            if self.nodes[i].surrogate_flip[s] == -1: # flip
                                temp_node = self.nodes[i].left_child
                                self.nodes[i].left_child = self.nodes[i].right_child
                                self.nodes[i].right_child = temp_node
            else:
                # MCR-
                if self.nodes[i].feature == var_idx:
                    for s in range(self.nodes[i].num_surrogates):
                        if self.nodes[i].surrogate_feature[s] != var_idx:
                            self.nodes[i].feature = self.nodes[i].surrogate_feature[s]
                            self.nodes[i].threshold = self.nodes[i].surrogate_threshold[s]
                            if self.nodes[i].surrogate_flip[s] == -1: # flip
                                temp_node = self.nodes[i].left_child
                                self.nodes[i].left_child = self.nodes[i].right_child
                                self.nodes[i].right_child = temp_node

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples, 
                          SIZE_t* surrogate_flip,
                          DOUBLE_t* surrogate_threshold,
                          SIZE_t* surrogate_feature, 
                          SIZE_t num_surrogates
                        ) nogil except -1:
        """Add a node to the tree.
        SIZE_t surrogate_flip[100]  # For surrogates it indicates if the condition needs to be flipped. (-1 if it does, 1 if not)
        DTYPE_t surrogate_threshold[100]  # 
        SIZE_t surrogate_feature[100]  # 
        SIZE_t num_surrogates

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """

        #with gil:
        #    print(f'num_surrogates for node_id: {self.node_count}: {num_surrogates}')
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.num_surrogates = -999999

       
            
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
            node.num_surrogates = num_surrogates

            for ig in range(num_surrogates):
                node.surrogate_flip[ig] = surrogate_flip[ig]
                node.surrogate_threshold[ig] = surrogate_threshold[ig]
                node.surrogate_feature[ig] = surrogate_feature[ig]


        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out
    
    cpdef np.ndarray predict_vim(self, object X, object permuted_vars, int mcr_type):
        """Predict target for X."""
        
        out = self._get_value_ndarray().take(self._apply_dense_surrogate(X,permuted_vars,mcr_type), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out
    
    cpdef np.ndarray predict_vim_via_ordering(self, object X, object permuted_vars, object mcr_ordering_pre, object mcr_ordering_others, object mcr_ordering_post):
        """Predict target for X."""
        
        out = self._get_value_ndarray().take(self._apply_dense_surrogate_via_mcr_ordering(X,permuted_vars,mcr_ordering_pre,mcr_ordering_others,mcr_ordering_post), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out
 


        

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)




    cpdef void print_tree(self, object col_names):

        cdef Node* node = self.nodes
        #with nogil:
        self.print_tree_lvl(0, node,col_names)


    cdef void print_tree_lvl(self, int lvl, Node* node, object col_names):
        
        if node.left_child == _TREE_LEAF:
            
            print( '{} {}'.format(' '*lvl*4, self._get_value_ndarray()[<SIZE_t>(node - self.nodes)] ))
            return

        if lvl == -1:
            pass
        else: 

            to_use = []
            for i in range(node.num_surrogates):
  
                if node.surrogate_flip[i] == 1:
                    op = '<='
                elif node.surrogate_flip[i] == -1:
                    op = '>'
                else:
                    op = '?'
                
                s = '({} {} {})'.format( col_names[node.surrogate_feature[i]], op, node.surrogate_threshold[i] )
                to_use.append(s)

            suros = ' '.join(to_use)

            st = 'f_idx: {} [{}] x <= {}'.format(col_names[node.feature], suros, node.threshold)
            print('{} {}'.format(' '*lvl*4, st) )

     
        # While node not a leaf
        if node.left_child != _TREE_LEAF:
            # ... and node.right_child != _TREE_LEAF:
            
            self.print_tree_lvl(lvl+1, &self.nodes[node.left_child],col_names)
            
            self.print_tree_lvl(lvl+1, &self.nodes[node.right_child],col_names)


    cdef inline np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out

    cdef inline np.ndarray _apply_dense_surrogate_via_mcr_ordering_orig2(self, object X, object permuted_vars, object mcr_ordering_pre, object mcr_ordering_others, object mcr_ordering_post):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))
        
        if not isinstance(mcr_ordering_pre, np.ndarray):
            raise ValueError("mcr_ordering should be in np.ndarray format, got %s"
                             % type(mcr_ordering_pre))
        
        if not isinstance(mcr_ordering_others, np.ndarray):
            raise ValueError("mcr_ordering should be in np.ndarray format, got %s"
                             % type(mcr_ordering_others))
        
        if not isinstance(mcr_ordering_post, np.ndarray):
            raise ValueError("mcr_ordering should be in np.ndarray format, got %s"
                             % type(mcr_ordering_post))


        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        
        cdef const SIZE_t[:] perm_vars = permuted_vars
        cdef SIZE_t n_permvars = len(permuted_vars)

        cdef const SIZE_t[:] mcr_order_pre = mcr_ordering_pre
        cdef SIZE_t n_mcr_order_pre = len(mcr_ordering_pre)

        cdef const SIZE_t[:] mcr_order_others = mcr_ordering_others
        cdef SIZE_t n_mcr_order_others = len(mcr_ordering_others)

        cdef const SIZE_t[:] mcr_order_post = mcr_ordering_post
        cdef SIZE_t n_mcr_order_post = len(mcr_ordering_post)

        cdef SIZE_t feature_to_use = 0
        cdef DTYPE_t threshold_to_use = 0
        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t mcr_idx = 0
        cdef int found = 0
        
        cdef int do_not_flip = 1 # 1 is don't flip, -1 is flip.  

        #print('Number of permuted variables: {}'.format(n_permvars))
        #print('MCR TYPE: {}'.format(c_mcr_type))

        #print('Using ordering based MCR.')

        with nogil:
            for i in range(n_samples):
                #with gil:
                #    print('\n========== CONSIDERING SAMPLE {} ================'.format(i))
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    #with gil:
                    #    print(f'Node has {node.num_surrogates} surrogates')
                  
                    do_not_flip = 1 
                    feature_to_use = node.feature
                    threshold_to_use = node.threshold

                    found = 0

                    ######## PRE. Search L-R through surrogates OR split
                    # for each variable in the order of use (L2R for MCR+ and MCR-, this ordering will be different in each case)
                    for io in range(n_mcr_order_pre):

                        # check if real split is this var
                        if feature_to_use == mcr_order_pre[io]:
                            found = -1
                            break
                        else:
                            # check all surrogates
                            for iga in range(node.num_surrogates):
                                if node.surrogate_feature[iga] == mcr_order_pre[io]:
                                    found = 1
                                    feature_to_use = node.surrogate_feature[iga] 
                                    threshold_to_use = node.surrogate_threshold[iga] 
                                    do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                                    break
                        
                    # OTHERS PART 1: 1st check all OTHERS to see if contains the actual split
                    for io in range(n_mcr_order_others):
                        if found != 0:
                            break
                        # check if real split is this var
                        if feature_to_use == mcr_order_others[io]:
                            found = -1
                            break
                        elif node.surrogate_feature[iga] == mcr_order_others[io]:
                            found = 1
                            feature_to_use = node.surrogate_feature[iga] 
                            threshold_to_use = node.surrogate_threshold[iga] 
                            do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                            break

                        
                    # POST search L-R through surrogates OR split
                    # for each variable in the order of use (L2R for MCR+ and MCR-, this ordering will be different in each case)
                    for io in range(n_mcr_order_post):
                        if found != 0:
                            break

                        # check if real split is this var
                        if feature_to_use == mcr_order_post[io]:
                            found = -1
                            break
                        else:
                            # check all surrogates
                            for iga in range(node.num_surrogates):
                                if node.surrogate_feature[iga] == mcr_order_post[io]:
                                    found = 1
                                    feature_to_use = node.surrogate_feature[iga] 
                                    threshold_to_use = node.surrogate_threshold[iga] 
                                    do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                                    break

                        if found != 0:
                            #with gil:
                            #    print(f'feature_to_use: {feature_to_use}')
                            #    print(f'threshold_to_use: {threshold_to_use}')
                            break

                    if do_not_flip == 1:
                        #with gil:
                            #if i == 0:
                            #print('Considering feature: {}] {} <= {}. If True: going LEFT.'.format(feature_to_use, X_ndarray[i, feature_to_use], threshold_to_use))
                        if X_ndarray[i, feature_to_use] <= threshold_to_use:
                            node = &self.nodes[node.left_child]
                        else:
                            node = &self.nodes[node.right_child]
                    else:
                        #with gil:
                            #if i == 0:
                            #print('Considering feature: {}] {} <= {}. If True: going RIGHT.'.format(feature_to_use, X_ndarray[i, feature_to_use], threshold_to_use))
                        if X_ndarray[i, feature_to_use] <= threshold_to_use:
                            node = &self.nodes[node.right_child]
                        else:
                            node = &self.nodes[node.left_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        
        #print('===> {}'.format(out))

        return out


    cdef inline np.ndarray _apply_dense_surrogate_via_mcr_ordering(self, object X, object permuted_vars, object mcr_ordering_pre, object mcr_ordering_others, object mcr_ordering_post):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))
        
        if not isinstance(mcr_ordering_pre, np.ndarray):
            raise ValueError("mcr_ordering should be in np.ndarray format, got %s"
                             % type(mcr_ordering_pre))
        
        if not isinstance(mcr_ordering_others, np.ndarray):
            raise ValueError("mcr_ordering should be in np.ndarray format, got %s"
                             % type(mcr_ordering_others))
        
        if not isinstance(mcr_ordering_post, np.ndarray):
            raise ValueError("mcr_ordering should be in np.ndarray format, got %s"
                             % type(mcr_ordering_post))


        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        
        cdef const SIZE_t[:] perm_vars = permuted_vars
        cdef SIZE_t n_permvars = len(permuted_vars)

        cdef const SIZE_t[:] mcr_order_pre = mcr_ordering_pre
        cdef SIZE_t n_mcr_order_pre = len(mcr_ordering_pre)

        cdef const SIZE_t[:] mcr_order_others = mcr_ordering_others
        cdef SIZE_t n_mcr_order_others = len(mcr_ordering_others)

        cdef const SIZE_t[:] mcr_order_post = mcr_ordering_post
        cdef SIZE_t n_mcr_order_post = len(mcr_ordering_post)

        cdef SIZE_t feature_to_use = 0
        cdef DTYPE_t threshold_to_use = 0
        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t mcr_idx = 0
        cdef int found = 0
        cdef int in_perm_set = 0
        
        cdef int do_not_flip = 1 # 1 is don't flip, -1 is flip.  

        #print('Number of permuted variables: {}'.format(n_permvars))
        #print('MCR TYPE: {}'.format(c_mcr_type))

        #print('Using ordering based MCR.')

        with nogil:
            for i in range(n_samples):
                #with gil:
                #    print('\n========== CONSIDERING SAMPLE {} ================'.format(i))
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    #with gil:
                    #    print(f'Node has {node.num_surrogates} surrogates')
                  
                    do_not_flip = 1 
                    feature_to_use = node.feature
                    threshold_to_use = node.threshold

                    found = 0

                    ######## PRE. Search L-R through surrogates OR split
                    # for each variable in the order of use (L2R for MCR+ and MCR-, this ordering will be different in each case)
                    for io in range(n_mcr_order_pre):

                        # check if real split is this var
                        if feature_to_use == mcr_order_pre[io]:
                            found = -1
                        else:
                            # check all surrogates
                            for iga in range(node.num_surrogates):
                                if node.surrogate_feature[iga] == mcr_order_pre[io]:
                                    found = 1
                                    feature_to_use = node.surrogate_feature[iga] 
                                    threshold_to_use = node.surrogate_threshold[iga] 
                                    do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip

                                    # check if in perm set
                                    # in_perm_set = 0
                                    # for pi in range(n_permvars):
                                    #     if feature_to_use == perm_vars[pi]:
                                    #         in_perm_set = 1

                                    # # if not in perm set
                                    # if in_perm_set == 0:
                                    #     feature_to_use = node.feature
                                    #     threshold_to_use = node.threshold
                                    #     do_not_flip = 1 

                                    break

                        if found != 0:
                            break
                        
                    # OTHERS: 1st check all OTHERS to see if contains the actual split
                    for io in range(n_mcr_order_others):
                        if found != 0:
                            break
                        # check if real split is this var
                        if feature_to_use == mcr_order_others[io]:
                            found = -1
                            break
                        # check all surrogates
                        for iga in range(node.num_surrogates):
                            if node.surrogate_feature[iga] == mcr_order_others[io]:
                                found = 1
                                feature_to_use = node.surrogate_feature[iga] 
                                threshold_to_use = node.surrogate_threshold[iga] 
                                do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                                
                                # check if in perm set
                                in_perm_set = 0
                                for pi in range(n_permvars):
                                    if feature_to_use == perm_vars[pi]:
                                        in_perm_set = 1

                                # if not in perm set
                                if in_perm_set == 0:
                                    feature_to_use = node.feature
                                    threshold_to_use = node.threshold
                                    do_not_flip = 1 
                                break
                            
                                
                        if found != 0:
                            break
                
                    

                    # POST search L-R through surrogates OR split
                    # for each variable in the order of use (L2R for MCR+ and MCR-, this ordering will be different in each case)
                    for io in range(n_mcr_order_post):
                        if found != 0:
                            break

                        # check if real split is this var
                        if feature_to_use == mcr_order_post[io]:
                            found = -1
                            break

                        # check all surrogates
                        for iga in range(node.num_surrogates):
                            if node.surrogate_feature[iga] == mcr_order_post[io]:
                                found = 1
                                feature_to_use = node.surrogate_feature[iga] 
                                threshold_to_use = node.surrogate_threshold[iga] 
                                do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                                
                                # check if in perm set
                                in_perm_set = 0
                                for pi in range(n_permvars):
                                    if feature_to_use == perm_vars[pi]:
                                        in_perm_set = 1

                                # if not in perm set
                                if in_perm_set == 0:
                                    feature_to_use = node.feature
                                    threshold_to_use = node.threshold
                                    do_not_flip = 1 

                                break

                        if found != 0:
                            break

                    if do_not_flip == 1:
                        #with gil:
                            #if i == 0:
                            #print('Considering feature: {}] {} <= {}. If True: going LEFT.'.format(feature_to_use, X_ndarray[i, feature_to_use], threshold_to_use))
                        if X_ndarray[i, feature_to_use] <= threshold_to_use:
                            node = &self.nodes[node.left_child]
                        else:
                            node = &self.nodes[node.right_child]
                    else:
                        #with gil:
                            #if i == 0:
                            #print('Considering feature: {}] {} <= {}. If True: going RIGHT.'.format(feature_to_use, X_ndarray[i, feature_to_use], threshold_to_use))
                        if X_ndarray[i, feature_to_use] <= threshold_to_use:
                            node = &self.nodes[node.right_child]
                        else:
                            node = &self.nodes[node.left_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        
        #print('===> {}'.format(out))

        return out


    cdef inline np.ndarray _apply_dense_surrogate(self, object X, object permuted_vars, int mcr_type):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        cdef SIZE_t c_mcr_type = mcr_type
        cdef const SIZE_t[:] perm_vars = permuted_vars
        cdef SIZE_t n_permvars = len(permuted_vars)
        cdef SIZE_t feature_to_use = 0
        cdef DTYPE_t threshold_to_use = 0
        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef int found = 0
        cdef int do_not_flip = 1 # 1 is don't flip, -1 is flip.  

        #print('Number of permuted variables: {}'.format(n_permvars))
        #print('MCR TYPE: {}'.format(c_mcr_type))

        with nogil:
            for i in range(n_samples):
                #with gil:
                #    print('\n========== CONSIDERING SAMPLE {} ================'.format(i))
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    #with gil:
                    #    print('-------------->{}'.format(node.num_surrogates))
                  
                    do_not_flip = 1 
                    feature_to_use = node.feature
                    threshold_to_use = node.threshold

                    found = 0
                    # TODO: Speed this up using hashmaps.
                    if c_mcr_type > 0: # MCR+
                    # force the use of permuted features

                        # If real split is in the perm set
                        for ig in range(n_permvars):
                            if feature_to_use == perm_vars[ig]:
                                found = 1
                                #with gil:
                                #    print('Split was already the var of interest: MCR+. USING: Feature: {}, Thresh: {}, Flip: {}'.format(feature_to_use, threshold_to_use,do_not_flip))
                                break
                        
                        # otherwise
                        if found == 0:
                            for iga in range(node.num_surrogates):
                            # search all of the surrogates
                                for ig in range(n_permvars):
                                # for the surrogate we are checking, does it match a var in the perm set?
                                    if node.surrogate_feature[iga] == perm_vars[ig]:
                                        found = 1
                                        break
                                if found == 1:
                                    feature_to_use = node.surrogate_feature[iga] 
                                    threshold_to_use = node.surrogate_threshold[iga] 
                                    do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                                    #with gil:
                                    #        print('Forcing use of surrogate: MCR+. Feature: {}, Thresh: {}, Flip: {}'.format(feature_to_use, threshold_to_use,do_not_flip))
                                    break

                        #if found == 0:
                        #    with gil:
                        #        print('Did not find an alternative, using var that is not the var of interest. Feature: {}, Thresh: {}, Flip: {}'.format(feature_to_use, threshold_to_use,do_not_flip))

                    if c_mcr_type < 0: # MCR-

                    # avoid the use of the permuted features
                        found = 0
                        # If real split is in the perm set
                        for ig in range(n_permvars):
                            if feature_to_use == perm_vars[ig]:
                                found = 1
                                break
                        
                        if found == 1:
                            # look for an alternative from surrogates that is not perm
                            for iga in range(node.num_surrogates):
                                found = 0
                                for ig in range(n_permvars):
                                    if node.surrogate_feature[iga] == perm_vars[ig]:
                                        found = 1
                                
                                if found == 0: # i.e. we found a surrogate that was not in the perm set use it instead
                                    feature_to_use = node.surrogate_feature[iga] 
                                    threshold_to_use = node.surrogate_threshold[iga] 
                                    do_not_flip = node.surrogate_flip[iga] # yes this is correct surrogate_flip[x] = 1 means don't flip
                                    found = 9999
                                    #with gil:
                                    #    print('AVOIDING the variables of interest. Using feature: {}, Thresh: {}, Flip: {}'.format(feature_to_use, threshold_to_use,do_not_flip))

                                    break
                        #if found < 99:
                            #with gil:
                            #    print('Did not find an alternative, using var that is not the var of interest. Feature: {}, Thresh: {}, Flip: {}'.format(feature_to_use, threshold_to_use,do_not_flip))
                   

                    if do_not_flip == 1:
                        #with gil:
                        #    if i == 0:
                        #        print('Considering feature: {}] {} <= {}. If True: going LEFT.'.format(feature_to_use, X_ndarray[i, feature_to_use], threshold_to_use))
                        if X_ndarray[i, feature_to_use] <= threshold_to_use:
                            node = &self.nodes[node.left_child]
                        else:
                            node = &self.nodes[node.right_child]
                    else:
                        #with gil:
                        #    if i == 0:
                        #        print('Considering feature: {}] {} <= {}. If True: going RIGHT.'.format(feature_to_use, X_ndarray[i, feature_to_use], threshold_to_use))
                        if X_ndarray[i, feature_to_use] <= threshold_to_use:
                            node = &self.nodes[node.right_child]
                        else:
                            node = &self.nodes[node.left_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        
        #print('===> {}'.format(out))

        return out



    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_sparse_csr(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out


    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cpdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray, # subtype
                                   <np.dtype> NODE_DTYPE, # descr
                                   1, #nd
                                   shape, #dims
                                   strides, #strides
                                   <void*> self.nodes, #data
                                   np.NPY_DEFAULT,  #flags
                                   None )# obj
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    def compute_partial_dependence(self, DTYPE_t[:, ::1] X,
                                   int[::1] target_features,
                                   double[::1] out):
        """Partial dependence of the response on the ``target_feature`` set.

        For each sample in ``X`` a tree traversal is performed.
        Each traversal starts from the root with weight 1.0.

        At each non-leaf node that splits on a target feature, either
        the left child or the right child is visited based on the feature
        value of the current sample, and the weight is not modified.
        At each non-leaf node that splits on a complementary feature,
        both children are visited and the weight is multiplied by the fraction
        of training samples which went to each child.

        At each leaf, the value of the node is multiplied by the current
        weight (weights sum to 1 for all visited terminal nodes).

        Parameters
        ----------
        X : view on 2d ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : view on 1d ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        out : view on 1d ndarray, shape (n_samples)
            The value of the partial dependence function on each grid
            point.
        """
        cdef:
            double[::1] weight_stack = np.zeros(self.node_count,
                                                dtype=np.float64)
            SIZE_t[::1] node_idx_stack = np.zeros(self.node_count,
                                                  dtype=np.intp)
            SIZE_t sample_idx
            SIZE_t feature_idx
            int stack_size
            double left_sample_frac
            double current_weight
            double total_weight  # used for sanity check only
            Node *current_node  # use a pointer to avoid copying attributes
            SIZE_t current_node_idx
            bint is_target_feature
            SIZE_t _TREE_LEAF = TREE_LEAF  # to avoid python interactions

        for sample_idx in range(X.shape[0]):
            # init stacks for current sample
            stack_size = 1
            node_idx_stack[0] = 0  # root node
            weight_stack[0] = 1  # all the samples are in the root node
            total_weight = 0

            while stack_size > 0:
                # pop the stack
                stack_size -= 1
                current_node_idx = node_idx_stack[stack_size]
                current_node = &self.nodes[current_node_idx]

                if current_node.left_child == _TREE_LEAF:
                    # leaf node
                    out[sample_idx] += (weight_stack[stack_size] *
                                        self.value[current_node_idx])
                    total_weight += weight_stack[stack_size]
                else:
                    # non-leaf node

                    # determine if the split feature is a target feature
                    is_target_feature = False
                    for feature_idx in range(target_features.shape[0]):
                        if target_features[feature_idx] == current_node.feature:
                            is_target_feature = True
                            break

                    if is_target_feature:
                        # In this case, we push left or right child on stack
                        if X[sample_idx, feature_idx] <= current_node.threshold:
                            node_idx_stack[stack_size] = current_node.left_child
                        else:
                            node_idx_stack[stack_size] = current_node.right_child
                        stack_size += 1
                    else:
                        # In this case, we push both children onto the stack,
                        # and give a weight proportional to the number of
                        # samples going through each branch.

                        # push left child
                        node_idx_stack[stack_size] = current_node.left_child
                        left_sample_frac = (
                            self.nodes[current_node.left_child].weighted_n_node_samples /
                            current_node.weighted_n_node_samples)
                        current_weight = weight_stack[stack_size]
                        weight_stack[stack_size] = current_weight * left_sample_frac
                        stack_size += 1

                        # push right child
                        node_idx_stack[stack_size] = current_node.right_child
                        weight_stack[stack_size] = (
                            current_weight * (1 - left_sample_frac))
                        stack_size += 1

            # Sanity check. Should never happen.
            if not (0.999 < total_weight < 1.001):
                raise ValueError("Total weight should be 1.0 but was %.9f" %
                                 total_weight)


