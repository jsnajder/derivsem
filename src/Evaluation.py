
from Models import *
from composes.semantic_space.space import Space
from composes.similarity.cos import CosSimilarity
from scipy.stats import pearsonr, spearmanr
import scipy as sp
from scipy import stats
from operator import itemgetter
from sklearn.cross_validation import KFold


def eval_correlation(y_predict, y_gold):
    cov = len(filter(lambda yp: yp > 0, y_predict))
    xs = filter(lambda y: y[1] != 0, zip(y_gold, y_predict))
    [y_gold_cov, y_predict_cov] = zip(*xs)
    r = pearsonr(y_gold, y_predict)[0]
    rho = spearmanr(y_gold, y_predict)[0]
    r_cov = pearsonr(y_gold_cov, y_predict_cov)[0]
    rho_cov = spearmanr(y_gold_cov, y_predict_cov)[0]
    return "r = %f, rho = %f, r_cov = %f, rho_cov = %f, cov = %d (%.2f%%)" % (r, rho, r_cov, rho_cov, cov, float(cov) / len(y_predict))


def check_pos(word, pos):
    try:
        return word.split('_')[1] == pos
    except IndexError:
        return False


# Filters space by POS
def space_pos_filter(space, pos):
    ix = []
    ws = []
    for i, w in enumerate(space.id2row):
        if check_pos(w, pos):
            ix.append(i)
            ws.append(w)
    m = space.cooccurrence_matrix[ix]
    rows = ws
    cols = space.id2column
    return Space(m, rows, cols)


# Gets nearest neighbors of a given vector (DISSECT has no function for that)
# If n_neighbors == None, returns all neighbors (sorted by cosine similarity)
def get_neighbors(vector, space, n_neighbors=5, pos=None):
    if pos is not None:
        space = space_pos_filter(space, pos)
    targets = space.id2row
    if n_neighbors is None:
        n_neighbors = len(targets)
    n_neighbors = min(n_neighbors, len(targets))
    sims_to_matrix = CosSimilarity().get_sims_to_matrix(vector, space.cooccurrence_matrix)
    sorted_perm = sims_to_matrix.sorted_permutation(sims_to_matrix.sum, 1)
    return [(space.id2row[i], sims_to_matrix[i, 0]) for i in sorted_perm[:n_neighbors]]


# Computes recall out of N (RooN); default N=5
# Returns the number of true positives and RooN
def score(model, test_pairs, n_neighbors=5, pos=None, verbose=False):
    hits = 0
    # TODO: so speed up, pos-filter space here and only once for the complete pattern
    for (base, derived) in test_pairs:
        neighbors = get_neighbors(model.predict(base, verbose), model.space, n_neighbors, pos)
        if verbose:
            print base, "=>", derived
            print neighbors
        if derived in map(itemgetter(0), neighbors):
            hits += 1
            if verbose:
                print "HIT!"
        else:
            if verbose:
                print "MISS"
        if verbose:
            print
    score = float(hits) / len(test_pairs)
    if verbose:
        print "=> Score: %d out of %d (%.2f%%)" % (hits, len(test_pairs), score)
    return hits, score


# Returns the mean reciprocal rank, by inspecting max_neighbors around the predicted vector. If max_neighbors==None,
# all vectors from the model's space are inspected. If the correct target is not among the neighbors of the predicted
# vector, the reciprocal rank is set to zero.
def mrr_score(model, test_pairs, max_neighbors=None, pos=None, verbose=False):
    if verbose:
        print("Computing MRR score on %d pairs" % len(test_pairs))
    scores = reciprocal_rank_scores(model, test_pairs, max_neighbors, pos, verbose)
    return sp.mean(scores)


def reciprocal_rank_scores(model, test_pairs, max_neighbors=None, pos=None, verbose=False):
    scores = []
    for (base, derived) in test_pairs:
        neighbors = get_neighbors(model.predict(base, verbose), model.space, max_neighbors, pos)
        rank = 0
        for i, (w, _) in enumerate(neighbors):
            if w == derived:
                rank = i + 1
                break
        if verbose:
            print("%s: correct target '%s' is at rank %d out of %d" % (base, derived, rank, len(neighbors)))
        reciprocal_rank = 0 if rank == 0 else 1 / float(rank)
        scores.append(reciprocal_rank)
    return sp.array(scores)


# Splits integer 'm' into 'n' balanced bins
def split_integer(n, m):
    r = 0
    xs = []
    for i in range(0, m):
        x = n / float(m)
        y = round(x + r)
        xs.append(int(y))
        r += x - y
    return xs


# Cross-validation indices
def cv_ixs(n, folds, shuffle=True, random_state=None):
    ns = split_integer(n, folds)
    i = 0
    ixs = []
    for j in ns:
        ixs.append(range(i, j + i))
        i += j
    if shuffle:
        sp.random.seed(random_state)
        zs = sp.random.permutation(n)
        return [list(zs[ix]) for ix in ixs]
    else:
        return ixs


# Computes RooN using cross-validation (default: 10 folds)
# Returns RooN and margin of error, by default at 95% confidence
def score_cv(model, pairs, folds=10, random_state=None, n_neighbors=5, shuffle=True,
             pos=None, verbose=False, conf_level=0.95, test_pairs_extra=None):
    # If test_pairs_extra != None, then the list of test_pair instances will be
    # divided into `folds` portions and each portion will be added to
    # one of the test set folds
    if folds == 'loocv':
        folds = len(pairs)
    kf = KFold(n=len(pairs), n_folds=folds, shuffle=shuffle, random_state=random_state)
    if test_pairs_extra is not None:
        test_extra_ix = cv_ixs(len(test_pairs_extra), folds, shuffle=shuffle, random_state=random_state)

    total_hits = 0
    score_list = []

    for i, (train_ix, test_ix) in enumerate(kf):
        train_pairs = pairs[train_ix]
        test_pairs = pairs[test_ix]
        if test_pairs_extra is not None:
            test_pairs = sp.vstack((test_pairs, test_pairs_extra[test_extra_ix[i]]))
        if verbose:
            print("\n=== Fold %d ===\n" % i)
            if test_pairs_extra is not None:
                print("Training on pairs %s and testing on %s+%s\n" % (train_ix, test_ix, test_extra_ix[i]))
            else:
                print("Training on pairs %s and testing on %s\n" % (train_ix, test_ix))
        model.fit(train_pairs, verbose)
        hits, sc = score(model, test_pairs, n_neighbors=n_neighbors, pos=pos, verbose=verbose)
        total_hits += hits
        score_list.append(sc)
    score_, score_margin = sample_mean(score_list, conf_level)
    if verbose:
        print("\n=> Score across all folds: %d out of %d (%.2f +- %.2f)" %
              (total_hits, len(test_pairs), score_, score_margin))
    return total_hits, score_, score_margin


def filter_array(X, f):
    ix = sp.array([f(x) for x in X])
    return X[ix]


# Computes the sample mean and the margin error
def sample_mean(xs, conf_level=0.95):
    mean = sp.mean(xs)
    stdev = sp.std(xs, ddof=1)
    N = len(xs)
    critical_value = - stats.t.ppf((1 - conf_level) / 2, N - 1)
    margin = critical_value * (stdev / sp.sqrt(N))
    return mean, margin
