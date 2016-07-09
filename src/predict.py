import pandas as pd
from composes.transformation.scaling.row_normalization import RowNormalization
from composes.utils import io_utils
import sys
from Data import filter_pairs, get_word_pairs, pattern_pos, partition_pairs
from Evaluation import reciprocal_rank, neighbors_avg_sim, derived_vector_norm, base_derived_sim
from Models import BaselineModel, AdditiveModel, LexfunModel


##############################################################################
# Data paths

proj_path = '/home/jan/b9/derivsem/'
src_path = proj_path + 'src/'
data_path = '/data/dsm/sdewac/'


##############################################################################


def prediction_features(partitioned_pairs_df, model, patterns=None, verbose=False):
    if patterns is not None:
        partitioned_pairs_df = partitioned_pairs_df[partitioned_pairs_df['pattern'].isin(patterns)]

    for pattern, pairs_df in partitioned_pairs_df.groupby('pattern'):
        print('Running on pattern %s with %d pairs' % (pattern, len(pattern)))
        train_pairs = get_word_pairs(filter_pairs(pairs_df, pattern, 0))
        model.fit(train_pairs, verbose=verbose)
        _, target_pos = pattern_pos(pattern)
        df = []
        for _, pair in pairs_df.iterrows():
            base = pair['word1']
            derived = pair['word2']
            print('\t %s %s' % (pair['word1'], pair['word2']))
            rr = reciprocal_rank(model, base, derived, pos=target_pos)
            ns = neighbors_avg_sim(model, base, pos=target_pos)
            vn = derived_vector_norm(model, base)
            bs = base_derived_sim(model, base)
            df.append(pd.Series({'avg_neighbors_sim': ns, 'derived_norm': vn, 'base_derived_sim': bs, 'rr': rr}))
        return pd.concat([partitioned_pairs_df, pd.concat(df, axis=1).T], axis=1)


##############################################################################
# Main

def main():

    partitioned_pairs_file = sys.argv[1]
    model_id = sys.argv[2]
    space_id = sys.argv[3]
    results_dir = sys.argv[4]

    partitioned_pairs_df = pd.read_csv(partitioned_pairs_file, sep=' ')

    space_file = {
        'cbow-w2': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w2.vsm.pkl',
        'cbow-w5': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w5.vsm.pkl',
        'cbow-w10': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w10.vsm.pkl',
        'ppmi': 'count-based/sdewac_2015-11-23/sdewac-mst.prepro.bow-c10k-w5.ppmi.matrix.pkl'
    }

    space = io_utils.load(data_path + space_file[space_id]).apply(RowNormalization(criterion='length'))

    models = {
        'baseline' : BaselineModel(space),
        'add' : AdditiveModel(space),
        'lexfun' : LexfunModel(space, learner='Ridge')
    }

    model = models[model_id]

    df = prediction_features(partitioned_pairs_df, model, verbose=False)

    df.to_pickle(results_dir + 'pairs-predictions-' + model_id + '-' + space_id + '.pkl')

    df.to_csv(results_dir + 'pairs-predictions-' - model_id + '-' + space_id + '.csv')

if __name__ == "__main__":
    main()


