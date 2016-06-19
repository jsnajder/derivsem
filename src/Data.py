import pandas as pd
import random


# Generates a list of 'n' indices, each indexing one of the len(proportions), according to given proportions.
# E.g., 50 [0.5, 0.3, 0.2]
def index_partitions(n, proportions, random_state=None):
    s = float(sum(proportions))
    ms = [int(round(p / s * n)) for p in proportions]
    xs = []
    for i, m in enumerate(ms):
        xs.extend([i] * m)
    random.seed(random_state)
    random.shuffle(xs)
    return xs[:n]


def partition_pairs(pairs_df, split, random_state=None):
    df = pd.DataFrame()
    ps = []
    for _, p in pairs_df.groupby('pattern'):
        n = len(p)
        ix = index_partitions(n, split, random_state=random_state)
        p.loc[:, 'partition'] = pd.Series(ix, index=p.index)
        ps.append(p)
    return pd.concat(ps)


def filter_pairs(pairs_df, pattern, partition=None):
    ix = pairs_df['pattern'] == pattern
    if partition is not None:
        ix &= pairs_df['partition'] == partition
    return pairs_df[ix]


def get_word_pairs(pairs_df):
    return pairs_df[['word1', 'word2']].values.tolist()


def pattern_pos(pattern): return pattern[1], pattern[2]