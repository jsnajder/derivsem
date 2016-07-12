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

def prediction_features(partitioned_pairs_df, model, patterns=None, verbose=False, pattern_map={}):
    def map_pattern(p):
        return pattern_map.get(p, p)

    partitioned_pairs_df['superpattern'] = partitioned_pairs_df.apply(lambda x: map_pattern(x['pattern']), axis=1)

    df = pd.DataFrame()

    for superpattern, pairs_df in partitioned_pairs_df.groupby('superpattern'):

        # Training an all patterns of a supergroup
        print('Running on superpattern "%s" with %d pairs' % (superpattern, len(pairs_df)))

        # Skip supergroup if none of the selected patterns in this superpattern group
        if (patterns is not None) and not (set(pairs_df['pattern']) & set(patterns)):
            print('Skipping this supergroup')
            continue

        pairs_train_df = pairs_df[pairs_df['partition'] == 0]
        print('Training on %d pairs...' % len(pairs_train_df))
        train_pairs = get_word_pairs(pairs_train_df)
        model.fit(train_pairs, verbose=verbose)

        # Filter selected patterns for testing
        if patterns is not None:
            pairs_filtered_df = pairs_df[pairs_df['pattern'].isin(patterns)]

        # Test on selected patterns
        print('Testing on %d pairs...' % len(pairs_filtered_df))
        for i, pair in pairs_filtered_df.iterrows():
            _, target_pos = pattern_pos(pair['pattern'])
            base = pair['word1']
            derived = pair['word2']
            print('\t %s %s' % (pair['word1'], pair['word2']))
            rr = reciprocal_rank(model, base, derived, pos=target_pos)
            ns = neighbors_avg_sim(model, base, pos=target_pos)
            vn = derived_vector_norm(model, base)
            bs = base_derived_sim(model, base)
            df = df.append(
                pd.Series({'avg_neighbors_sim': ns, 'derived_norm': vn, 'base_derived_sim': bs, 'rr': rr}, name=i))

    return partitioned_pairs_df.join(df)


##############################################################################
# Main

def main():

    partitioned_pairs_file = sys.argv[1]
    patterns_file = sys.argv[2]
    model_id = sys.argv[3]
    space_id = sys.argv[4]
    pattern_map_file = sys.argv[5]
    results_file = sys.argv[6]

    partitioned_pairs_df = pd.read_csv(partitioned_pairs_file, index_col=0)

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

    if patterns_file == 'None':
        patterns = None
    else:
        patterns = []
        with open(patterns_file) as f:
            for l in f.read().splitlines():
                patterns += l.split(' ')

    if pattern_map_file == 'None':
        pattern_map = {}
    else:
        pattern_map = {}
        with open(pattern_map_file) as f:
            for l in f.read().splitlines():
                xs = l.split(' ')
                superpattern = xs[0]
                for p in xs[1:]:
                    pattern_map[p] = superpattern

    df = prediction_features(partitioned_pairs_df, model, patterns, verbose=False, pattern_map=pattern_map)

    df.to_pickle(results_file + '.pkl')

    df.to_csv(results_file + '.csv')

if __name__ == "__main__":
    main()


