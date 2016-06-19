#!/usr/bin/env/python
# -*- coding: utf-8 -*-
#
# usage: python EvalPatterns.py [ppmi|cbow] <pairs.txt> <pattern_set_string> <results_dir>

import sys
import scipy as sp
import pandas as pd
from composes.semantic_space.space import Space
from composes.utils import io_utils
from composes.transformation.scaling.row_normalization import RowNormalization

from Models import *


##############################################################################

def pattern_pos(pattern): return (pattern[1], pattern[2])

# For a given pattern, fetches rows from a dataframe that satisfy the given poysemy and invCL thresholds.
# Returns two dataframes: one containing rows that satisfy the conditions and one containing those that don't.
def partition_pairs(pairs_df, pattern, polysemy_threshold=None, invCL_threshold=None, only_pairs=False):
    
    def get_pairs(df): return sp.array(df[['word1','word2']])
    
    ix0 = sp.logical_and(pairs_df.polysemy <= polysemy_threshold if polysemy_threshold != None else True,
                         pairs_df.invCL >= invCL_threshold if invCL_threshold != None else True)
    ix1 = sp.logical_and(pairs_df.pattern == pattern, ix0)
    ix2 = sp.logical_and(pairs_df.pattern == pattern, ~ix0)
    
    if only_pairs:
        return get_pairs(pairs_df[ix1]), get_pairs(pairs_df[ix2])
    else:
        return pairs_df[ix1], pairs_df[ix2]

def median_invCL(pairs_df, pattern):
    return pairs_df[pairs_df.pattern == pattern]['invCL'].median()

def eval_pattern(space, pairs_df, pattern, folds=10, random_state=None, verbose=False):

    models = [
        ('Baseline', BaselineModel(space)), 
        ('Additive', AdditiveModel(space)),
        ('AdditiveExemplar', AdditiveExemplarModel(space))] + \
        [('CluAdditive (DiffVectors, kmeans, k=%d, BasePredictSim)' % k, 
         ClusterAdditiveModel(space, clustering_instance='DiffVector', clustering='kmeans', n_clusters=k, cluster_select='BasePredictSim', random_state=random_state)) 
         for k in range(2,6)] + \
        [('CluAdditive (BaseWord, kmeans, k=%d, BasePredictSim)' % k, 
         ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmeans', n_clusters=k, cluster_select='BasePredictSim', random_state=random_state)) 
         for k in range(2,6)] + \
        [('CluAdditive (BaseWord, kmeans, k=%d, BaseClusterSim)' % k, 
         ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmeans', n_clusters=k, cluster_select='BaseClusterSim', random_state=random_state)) 
         for k in range(2,6)]

#    models = [
#        ('Baseline', BaselineModel(space)), 
#        ('Additive', AdditiveModel(space)),
#        ('AdditiveExemplar', AdditiveExemplarModel(space))] + \
#        [('CluAdditive (DiffVectors, kmeans, k=%d, BasePredictSim)' % k, 
#         ClusterAdditiveModel(space, clustering_instance='DiffVector', clustering='kmeans', n_clusters=k, cluster_select='BasePredictSim', random_state=random_state)) 
#         for k in range(2,6)] + \
##        [('CluAdditive (DiffVectors, kmedoids, k=%d, BasePredictSim)' % k, 
##         ClusterAdditiveModel(space, clustering_instance='DiffVector', clustering='kmedoids', n_clusters=k, cluster_select='BasePredictSim', random_state=random_state)) 
##         for k in range(2,6)] + \
#        [('CluAdditive (BaseWord, kmeans, k=%d, BasePredictSim)' % k, 
#         ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmeans', n_clusters=k, cluster_select='BasePredictSim', random_state=random_state)) 
#         for k in range(2,6)] + \
#        #[('CluAdditive (BaseWord, kmedoids, k=%d, BasePredictSim)' % k, 
#        # ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmedoids', n_clusters=k, cluster_select='BasePredictSim', random_state=random_state)) 
#        # for k in range(2,6)] + \
#        [('CluAdditive (BaseWord, kmeans, k=%d, BaseClusterSim)' % k, 
#         ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmeans', n_clusters=k, cluster_select='BaseClusterSim', random_state=random_state)) 
#         for k in range(2,6)] #+ \
#        #[('CluAdditive (BaseWord, kmedoids, k=%d, BaseClusterSim)' % k, 
#        # ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmedoids', n_clusters=k, cluster_select='BaseClusterSim', random_state=random_state)) 
#        # for k in range(2,6)]            

    invCL_median = median_invCL(pairs_df, pattern)

    pairs_all, _ = partition_pairs(pairs_df, pattern, only_pairs=True)
    pairs_mono1, pairs_mono0 = partition_pairs(pairs_df, pattern, polysemy_threshold=1, only_pairs=True)
    pairs_incl1, pairs_incl0 = partition_pairs(pairs_df, pattern, invCL_threshold=invCL_median, only_pairs=True)
    pairs_monoincl1, pairs_monoincl0 = partition_pairs(pairs_df, pattern, polysemy_threshold=1, invCL_threshold=invCL_median, only_pairs=True)

    _, deriv_pos = pattern_pos(pattern)

    data = [
        ('All', pairs_all, None),
        ('Mono', pairs_mono1, None),
        ('Incl', pairs_incl1, None),
        ('MonoIncl', pairs_monoincl1, None),
        ('Mono', pairs_mono1, pairs_mono0),
        ('Incl', pairs_incl1, pairs_incl0),
        ('MonoIncl', pairs_monoincl1, pairs_monoincl0)]

    model_names = [n for n, _ in models]
    data_names = ['%s (%s:%d+%d)' % (pattern, pairs_name, len(pairs_train), 
                                     len(pairs_extra_test) if pairs_extra_test != None else 0)
                  for pairs_name, pairs_train, pairs_extra_test in data]
    scores_df = pd.DataFrame(index=model_names, columns=data_names)
    
    for data_name, (_, pairs_train, pairs_extra_test) in zip(data_names, data):
        if verbose:
            print('Data: %s' % data_name)
        for model_name, model in models:
            _, rof, rof_error = score_cv(model, pairs_train, test_pairs_extra=pairs_extra_test,
                                         pos=deriv_pos, folds=folds, random_state=random_state)
            scores_df[data_name][model_name] = '%.3f ± %.2f' % (rof, rof_error)
            if verbose:
                print('  %s: %.3f ± %.2f' % (model_name, rof, rof_error))
    
    return scores_df

def eval_pattern2(space, pairs_df, pattern, folds=10, random_state=None, verbose=False):
    
    models = [
        ('Baseline', BaselineModel(space)),
        ('Additive', AdditiveModel(space)),
        ('Clustering', ClusterAdditiveModel(space, clustering_instance='BaseWord', clustering='kmeans', n_clusters='AIC', cluster_select='BaseClusterSim', random_state=42))]
    
    _, deriv_pos = pattern_pos(pattern)
    
    invCL_median = median_invCL(pairs_df, pattern)
    
    pairs_all, _ = partition_pairs(pairs_df, pattern, only_pairs=True)
    pairs_incl1, pairs_incl0 = partition_pairs(pairs_df, pattern, invCL_threshold=invCL_median, only_pairs=True)
    data = [
        ('All', pairs_all, None), 
        ('Inc', pairs_incl1, pairs_incl0)]
    
    model_names = [n for n, _ in models]
    data_names = [n for n, _, _ in data]
    result = pd.DataFrame(index=['n_pairs'] + model_names, columns=data_names)
    
    for model_name, model in models:
        for data_name, pairs_train, pairs_extra in data:
            if model_name == 'Baseline' and data_name == 'Inc': continue
            hits, _, _ = score_cv(model, pairs_train, test_pairs_extra=pairs_extra, pos=deriv_pos, folds=folds, random_state=random_state)
            result[data_name][model_name] = hits
            if verbose: print('%s: %d' % (model_name, hits))
    
    result['All']['n_pairs'] = len(pairs_all)
    result['Inc']['n_pairs'] = len(pairs_incl1)
    
    return result

##############################################################################

def main():

  data_path = "/data/dsm/sdewac/"
  
  model = sys.argv[1]
  pairs = sys.argv[2]
  pattern_set = sys.argv[3]
  results_dir = sys.argv[4]
  
  pairs_df = pd.read_csv(pairs, sep=' ')
  
  model_file = {
    'cbow-w2': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w2.vsm.pkl',
    'cbow-w5': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w5.vsm.pkl',
    'cbow-w10': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w10.vsm.pkl',
    'ppmi': 'count-based/sdewac_2015-11-23/sdewac-mst.prepro.bow-c10k-w5.ppmi.matrix.pkl'
  }
  
  space = io_utils.load(data_path + model_file[model])
  space = space.apply(RowNormalization(criterion = 'length'))
  
  patterns = pd.unique(pairs_df['pattern'])
  
  writer = pd.ExcelWriter(results_dir + '/eval-' + model + '-' + pattern_set + '.xlsx')
  for pattern in patterns:
      df = eval_pattern(space, pairs_df, pattern, folds=10, random_state=42, verbose=True)
      df.to_excel(writer, pattern)
      writer.save()

if __name__ == "__main__":
    main()
