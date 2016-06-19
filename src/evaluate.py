# coding: utf-8

# <h1>Ensemble model</h1>
#
#
# Version 1, 17 Jun 2016<br>
# Jan Šnajder

import pandas as pd
from composes.transformation.scaling.row_normalization import RowNormalization
from composes.utils import io_utils

from Data import filter_pairs, get_word_pairs, pattern_pos
from Models import *


##############################################################################
# Data paths

proj_path = '/home/jan/b9/derivsem/'
src_path = proj_path + 'src/'
data_path = '/data/dsm/sdewac/'


##############################################################################
# Spaces


space = {}

# CBOW

model_file = 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w2.vsm.pkl'
space['cbow-w2'] = io_utils.load(data_path + model_file).apply(RowNormalization(criterion='length'))
model_file = 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w5.vsm.pkl'
space['cbow-w5'] = io_utils.load(data_path + model_file).apply(RowNormalization(criterion='length'))
model_file = 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w10.vsm.pkl'
space['cbow-w10'] = io_utils.load(data_path + model_file).apply(RowNormalization(criterion='length'))

# Count-based

model_file = 'count-based/sdewac_2015-11-23/sdewac-mst.prepro.bow-c10k-w5.ppmi.matrix.pkl'
space['ppmi'] = io_utils.load(data_path + model_file).apply(RowNormalization(criterion='length'))

# TODO: COW model

##############################################################################
# Models

model = {}
for name, s in space.items():
    model['baseline-' + name] = BaselineModel(s)
    model['add-' + name] = AdditiveModel(s)
    model['lexfun-' + name] = LexfunModel(s, learner='Ridge')


##############################################################################


def evaluate(partitioned_pairs_df, models_dict, patterns=None):
    if patterns is not None:
        partitioned_pairs_df = partitioned_pairs_df[partitioned_pairs_df['pattern'].isin(patterns)]
    for model_name, model in models_dict.items():
        dfs = []
        for pattern, pairs_df in partitioned_pairs_df.groupby('pattern'):
            print model_name, pattern
            train_pairs = get_word_pairs(filter_pairs(pairs_df, pattern, 0))
            model.fit(train_pairs)
            _, target_pos = pattern_pos(pattern)
            scores_test = reciprocal_rank_scores(model, get_word_pairs(pairs_df), pos=target_pos)
            pairs_df.loc[:, 'baseline-cbow-w5'] = pd.Series(scores_test, index=pairs_df.index)
            dfs.append(pairs_df)
        return pd.concat(dfs)


##############################################################################
# Main