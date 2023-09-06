from io import StringIO
import itertools

import numpy as np
from prody import parsePDBStream, superpose

def compute_scores(x, y, mask=None, slice_output=slice(None)):
    xp = x.reshape(-1, 3)
    yp = y.reshape(-1, 3)
    if mask is not None:
        _, T = superpose(x[mask].reshape(-1, 3), y[mask].reshape(-1, 3))
        xp = T.apply(xp)
    else:
        xp, T = superpose(xp, yp)
    if mask is None:
        mask = np.ones(x.shape[:-2])[..., None]
    mask = mask[slice_output]
    xp = xp.reshape(x.shape)[slice_output].reshape(-1, 3)
    yp = yp.reshape(y.shape)[slice_output].reshape(-1, 3)
    rmsd = np.sqrt((((xp - yp) ** 2).sum(axis=-1) * mask).sum() / np.maximum(mask.sum(), 1e-6))
    dist = np.sqrt(((xp - yp) ** 2).sum(axis=-1) * mask)
    dist0 = 1.24 * (xp.shape[0] - 15) ** (1 / 3) - 1.8
    tm = (1 / (1 + (dist / dist0) ** 2)).sum() / np.maximum(mask.sum(), 1e-6)
    return rmsd, tm

def compute_scores_permuted(predicted, structure, chain_index, mask=None):
    chain_index = chain_index
    rmsd = np.inf
    tm = 0.0
    for permutation, m in permute(structure, mask, chain_index):
        new_rmsd, new_tm = compute_scores(predicted, permutation, mask=m)
        if new_rmsd < rmsd:
            rmsd = new_rmsd
            tm = new_tm
    return rmsd, tm
    
def permute(structure: np.ndarray[np.float32], mask, chain_index: np.ndarray[np.int32]):
    unique, count = np.unique(chain_index, return_counts=True)
    start_end = np.cumsum(np.concatenate((np.array([0], dtype=np.int32), count), axis=0), axis=0)
    start = start_end[:-1]
    end = start_end[1:]
    if len(unique) <= 1:
        yield structure, mask
    else:
        print("PERMU", unique)
        for perm in itertools.permutations(unique):
            permutation_index = np.concatenate([
                np.arange(start[i], end[i], dtype=np.int32)
                for i in perm
            ])
            m = mask
            if mask is not None:
                m = mask[permutation_index]
            yield structure[permutation_index], m
