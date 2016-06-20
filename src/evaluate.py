# coding: utf-8

# <h1>Ensemble model</h1>
#
#
# Version 1, 17 Jun 2016<br>
# Jan Šnajder

import pandas as pd
from composes.transformation.scaling.row_normalization import RowNormalization
from composes.utils import io_utils
import sys
from Data import filter_pairs, get_word_pairs, pattern_pos, partition_pairs
from Evaluation import reciprocal_rank_scores


##############################################################################
# Data paths

proj_path = '/home/jan/b9/derivsem/'
src_path = proj_path + 'src/'
data_path = '/data/dsm/sdewac/'


##############################################################################


def evaluate(partitioned_pairs_df, models_dict, patterns=None, verbose=False):
    if patterns is not None:
        partitioned_pairs_df = partitioned_pairs_df[partitioned_pairs_df['pattern'].isin(patterns)]
    for model_name, model in models_dict.items():
        dfs = []
        for pattern, pairs_df in partitioned_pairs_df.groupby('pattern'):
            print model_name, pattern
            train_pairs = get_word_pairs(filter_pairs(pairs_df, pattern, 0))
            model.fit(train_pairs, vebose=False)
            _, target_pos = pattern_pos(pattern)
            scores_test = reciprocal_rank_scores(model, get_word_pairs(pairs_df), pos=target_pos, verbose=verbose)
            pairs_df.loc[:, 'baseline-cbow-w5'] = pd.Series(scores_test, index=pairs_df.index)
            dfs.append(pairs_df)
        return pd.concat(dfs)


##############################################################################
# Main

def main():

    pairs_file = sys.argv[1]
    space = sys.argv[2]
    results_dir = sys.argv[3]

    pairs_df = pd.read_csv(pairs_file, sep=' ')

    space_file = {
        'cbow-w2': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w2.vsm.pkl',
        'cbow-w5': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w5.vsm.pkl',
        'cbow-w10': 'cbow/cbow_300dim_hs0/sdewac.300.cbow.hs0.w10.vsm.pkl',
        'ppmi': 'count-based/sdewac_2015-11-23/sdewac-mst.prepro.bow-c10k-w5.ppmi.matrix.pkl'
    }

    space = io_utils.load(data_path + space_file[space]).apply(RowNormalization(criterion='length'))

    split = [0.5, 0.3, 0.2]
    partitioned_pairs_df = partition_pairs(pairs_df, split, random_state=42)


    df = evaluate(partitioned_pairs_df, verbose=False)

    df.to_pickle(results_dir + space + '.pkl')

    writer = pd.ExcelWriter(results_dir + space + '.xlsx')
    df.to_excel(writer, space)
    writer.save()

if __name__ == "__main__":
    main()
