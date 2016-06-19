
# coding: utf-8

# <h1>Polysemy in Derivational Models</h1>
# 
# Version 10, 15 Jan 2016<br>
# Jan Šnajder

from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.composition.lexical_function import LexicalFunction
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.utils.regression_learner import RidgeRegressionLearner
import scipy as sp
from sklearn import mixture
from sklearn.cluster import KMeans
from operator import itemgetter
import functools
from scipy.sparse import csr_matrix
from k_medoids import KMedoids
import warnings


##############################################################################
# Helpers


def conjunction(*conditions):
    return functools.reduce(sp.logical_and, conditions)


# Cosine distance that works with both 'matrix' (dense) and 'csr_matrix'
# (sparse) types (unlike cosine distance from 'scipy.spatial.distance')
def my_dot(v1, v2): return v1.dot(v2.T)


def my_norm(v): return sp.sqrt(my_dot(v, v)[0, 0])


def my_cosine(v1, v2): return my_dot(v1, v2)[0, 0] / (my_norm(v1) * my_norm(v2))


def my_cosine_dist(v1, v2): return 1 - my_cosine(v1, v2)


def get_row_dense(space, word):
    v = space.get_row(word).mat
    if isinstance(v, sp.matrix):
        return v
    elif isinstance(v, csr_matrix):
        return v.todense()
    else:
        raise NameError('get_row_dense: unknown data type')


##############################################################################


class Model(object):
    
    def __init__(self, space):
        self.space = space
        
    def fit(self, train_pairs, verbose=False): return
    
    def predict(self, base, verbose=False): return


##############################################################################
# Baseline model

# The baseline model simply predicts the base vector
class BaselineModel(Model):
    def predict(self, base, verbose=False): 
        return self.space.get_row(base)


##############################################################################
# Additive model


class AdditiveModel(Model):
    
    diff_vector = None
    
    def fit(self, train_pairs, verbose=False):
        if len(train_pairs) == 0:
            raise NameError('Error: Train set is empty')
            # warnings.warn('fit: train set is empty, defaulting to baseline', UserWarning)
            # set difference vector to empty vector
            # self.diff_vector = DenseMatrix(sp.zeros(1))
        else:
            if verbose:
                print 'fit: Computing the diff vector across %d pairs' % (len(train_pairs))
            (_, n) = sp.shape(self.space.cooccurrence_matrix)
            if isinstance(self.space.cooccurrence_matrix, DenseMatrix):
                self.diff_vector = DenseMatrix(sp.zeros(n))
            else:
                self.diff_vector = SparseMatrix(sp.zeros(n))
            for (base, derived) in train_pairs:
                diff = self.space.get_row(derived) - self.space.get_row(base)
                self.diff_vector += diff
            self.diff_vector /= len(train_pairs)
        
    def predict(self, base, verbose=False):
        if self.diff_vector is None:
            raise NameError('Error: Model has not yet been trained')
        return self.space.get_row(base) + self.diff_vector


##############################################################################
# Lexfun model
# This uses the DISSECT's LexicalFunction class, which turned out to be a
# bit awkward as the APIs are quite different


class LexfunModel(Model):

    lexfun = None

    def __init__(self, space, learner='LeastSquares', param=None):
        # super(LexfunModel, self).__init__(space)
        Model.__init__(self, space)
        if learner == 'Ridge':
            # If param==None, generalized CV will be performed within standard param range
            learner = RidgeRegressionLearner(param)
        elif learner == 'LeastSquares':
            learner = LstsqRegressionLearner()
        else:
            raise NameError("No such learner: %s" % learner)
        self.lexfun = LexicalFunction(learner=learner)

    def fit(self, train_pairs, verbose=False):
        if len(train_pairs) == 0:
            raise NameError('Error: Train set is empty')
        else:
            if verbose:
                print 'fit: Fitting a lexfun model on %d pairs' % (len(train_pairs))
            # LexicalFunction class is designed to be run on a dataset with different function words (==patterns).
            # We use a dummy function word here.
            train_pairs_ext = [('dummy', base, derived) for (base, derived) in train_pairs]
            self.lexfun.train(train_pairs_ext, self.space, self.space)

    def predict(self, base, verbose=False):
        if self.lexfun is None:
            raise NameError('Error: Model has not yet been trained')
        composed_space = self.lexfun.compose([('dummy', base, 'derived')], self.space)
        return composed_space.get_row('derived')


##############################################################################
# Exemplar model


class AdditiveExemplarModel(Model):
    
    base_derived_list = None
    
    def fit(self, train_pairs, verbose=False):
        self.base_derived_list = []
        if verbose:
            print('fit: Storing %d train pairs' % (len(train_pairs)))
        for (base, derived) in train_pairs:
            base_v = self.space.get_row(base).mat
            diff = self.space.get_row(derived).mat - base_v
            self.base_derived_list.append((base, base_v, diff))

    def predict(self, base, verbose=False):
        if self.base_derived_list is None:
            raise NameError('Error: Model has not yet been trained')
        # Find the base vector that is most similar to input base
        v_base = self.space.get_row(base).mat
        i = argmax(self.base_derived_list, lambda (b, v, d): my_cosine(v, v_base))
        b, _, d = self.base_derived_list[i]
        if verbose:
            print('predict: Predicting derivation of %s with diff vector of base %s' % (base, b))
        return v_base + d


##############################################################################
# Clustering + additive model


# Returns difference vectors and diff vectors centroid
# as numpy arrays (rows) or as DISSECT DenseMatrix
def get_diff_vectors(space, pairs, denseMatrix=False):
  
    (_, n) = sp.shape(space.cooccurrence_matrix)
    diff_vectors = sp.empty((0, n), float)
  
    for (base, derived) in pairs:
        diff_vector = get_row_dense(space, derived) - get_row_dense(space, base)
        diff_vectors = sp.vstack((diff_vectors, diff_vector))
  
    diff_centroid = sp.mean(diff_vectors, axis=0)
  
    if denseMatrix:
        return DenseMatrix(diff_vectors), DenseMatrix(diff_centroid)
    else:
        return diff_vectors, diff_centroid


def get_base_vectors(space, pairs):
    (_, n) = sp.shape(space.cooccurrence_matrix)
    vectors = sp.empty((0, n), float)
    for (w1, _) in pairs:
        vector = get_row_dense(space, w1)
        vectors = sp.vstack((vectors, vector))
    return vectors


def argmax(xs, f):
    index, _ = max(enumerate(xs), key=lambda x: f(itemgetter(1)(x)))
    return index


def argmin(xs, f):
    index, _ = min(enumerate(xs), key=lambda x: f(itemgetter(1)(x)))
    return index


def avg_neighbors_sim(vect, space, n_neighbors=5, pos=None):
    return sp.mean([sim for _, sim in get_neighbors(vect, space, n_neighbors=n_neighbors, pos=pos)])


class ClusterAdditiveModel(Model):
    
    n_clusters = None
    _n_clusters = None
    random_state = None
    cluster_select = None
    models = None
    clustering = None
    cluster_assignments = None
    clustering_instance = None
    clusters_ = None
    
    def __init__(self, space, n_clusters='BIC', 
                 clustering_instance='DiffVector', 
                 cluster_select='BasePredictSim', 
                 clustering='kmeans', random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_select = cluster_select
        self.clustering = clustering
        self.clustering_instance = clustering_instance
        Model.__init__(self, space)
        # super(ClusterAdditiveModel, self).__init__(space)
        
    def opt_n_clusters(self, X, criterion='BIC', n_clusters_range=(2, 6)):
        best_K = 1
        min_score = float('inf')
        for K in range(n_clusters_range[0], n_clusters_range[1] + 1):
            g = mixture.GMM(n_components=K, random_state=self.random_state)
            g.fit(X)
            score = g.aic(X) if criterion == 'AIC' else g.bic(X)
            # print('%d: %d' % (K, score))
            if score < min_score:
                min_score = score
                best_K = K
        return best_K
    
    def fit(self, train_pairs, verbose=False):
        if self.clustering_instance == 'DiffVector':
            # Clustering of difference vectors
            X, _ = get_diff_vectors(self.space, train_pairs)
        elif self.clustering_instance == 'BaseWord':
            # Clustering of basis words
            X = get_base_vectors(self.space, train_pairs)
        else:
            raise NameError('Cluster instance can be either \'Diffvector\' or \'BaseWord\'')
        # Optimize number of clusters
        if self.n_clusters == 'BIC' or self.n_clusters == 'AIC':
            self._n_clusters = self.opt_n_clusters(X, criterion=self.n_clusters)
        else:
            self._n_clusters = self.n_clusters
        if verbose:
            print('train: Clustering %d pairs into %d clusters' % (len(train_pairs), self._n_clusters))
        # Choose algorithm
        if self.clustering == 'gmm':
            c = mixture.GMM(n_components=self._n_clusters, covariance_type='tied', random_state=self.random_state) 
        elif self.clustering == 'kmeans':
            c = KMeans(n_clusters=self._n_clusters, random_state=self.random_state)
        elif self.clustering == 'kmedoids':
            c = KMedoids(n_clusters=self._n_clusters, random_state=self.random_state, distance_metric='cosine')
        else:
            raise NameError('Unknown clustering algorithm') 
        # Fit the model
        c.fit(X)
        self.clusters_ = c
        # Assign clusters to pairs
        ix = c.predict(X)
        self.cluster_assignments = ix
        ms = []
        # Train separate linear models
        for k in range(0, self._n_clusters):
            train_pairs_k = train_pairs[ix == k]
            if len(train_pairs_k) == 0:
                warnings.warn('fit: cluster %d is empty' % k, UserWarning)
            if verbose:
                print 'fit: Computing the diff vector for cluster %d containing %d pairs' % (k, len(train_pairs_k))
            m = AdditiveModel(self.space)
            m.fit(train_pairs_k)
            ms.append(m)
        self.models = ms
    
    def predict_with(self, base, cluster_ix):
        if self.models is None:
            raise NameError('Error: Model has not yet been trained')
        else:
            return self.models[cluster_ix].predict(base)
    
    def predict(self, base, verbose=False):
        if self.cluster_select == 'BaseClusterSim':
            # Cluster selection is based on base-cluster similarity
            if self.clustering_instance != 'BaseWord':
                raise NameError('Cluster instance must be set to \'BaseWord\' to use \'CentroidSim\'')
            # Chose the closest cluster to the base
            v_base = self.space.get_row(base)
            k = self.clusters_.predict(v_base.mat)
        else:
            # Cluster selection is based on the predicted vector
            # First, compute predictions with each difference vector
            if self._n_clusters is None:
                raise NameError('Error: Model has not yet been trained')
            pred = []
            for k in range(0, self._n_clusters):
                pred.append(self.predict_with(base, k))
            # Now, choose a cluster based on cluster_select strategy
            if self.cluster_select == 'VectorLength':
                # Minimizing the L2-norm of the predicted vector
                k = argmin(pred, lambda v: sp.linalg.norm(v.mat))
            elif self.cluster_select == 'BasePredictSim':
                # Maximizing the cosine similarity to the base vector
                v_base = self.space.get_row(base)
                k = argmax(pred, lambda v: my_cosine(v.mat, v_base.mat))
            elif self.cluster_select == 'AvgNeighborSim':
                k = argmax(pred, lambda v: avg_neighbors_sim(v, self.space))
            else:
                raise NameError('Error: Undefined cluster_select strategy')
        if verbose: 
            print 'predict: predicting derivation of %s using diff vector of cluster %d' % (base, k)
        # Finally, predict using the diff vector from the chosen cluster
        return self.predict_with(base, k)


