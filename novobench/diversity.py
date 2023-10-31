# copyright (c) 2023 Michael Jendrusch, EMBL
# AlphaDesign: A de novo protein design framework based on AlphaFold
# Michael Jendrusch, Alessio Ling Jie Yang, Elisabetta Cacace, Jacob Bobonis,
# Carlos Geert Pieter Voogdt, Athanasios Typas, Jan O. Korbel, and S. Kashif Sadiq

from argparse import ArgumentParser
import os

from typing import Optional, Tuple

import numpy as np

from alphafold.common.protein import from_pdb_string

from novobench.analysis.alignment import compute_scores_permuted

def compute_distances(pdb_path: str,
                      out_path: str,
                      slice_pdb: Optional[str] = "none") -> Tuple[str, str, np.ndarray]:
    slice_pdb = parse_slice(slice_pdb)
    pdb_files = sorted(os.listdir(pdb_path))
    rmsd_array = np.zeros((len(pdb_files), len(pdb_files)), dtype=np.float32)
    tm_array = np.ones((len(pdb_files), len(pdb_files)), dtype=np.float32)
    names = np.array([".".join(pdb.split(".")[:-1]) for pdb in pdb_files])
    for idx, pdb_1 in enumerate(pdb_files):
        for idy, pdb_2 in enumerate(pdb_files[idx + 1:]):
            idy += 1 + idx
            structure_1, _, _, chain_index = read_pdb(os.path.join(pdb_path, pdb_1))
            structure_2, _, _, chain_index = read_pdb(os.path.join(pdb_path, pdb_2))
            mask = np.ones(structure_1.shape[0], dtype=np.bool_)
            rmsd, tm = compute_scores_permuted(structure_1[:, 1], structure_2[:, 1], chain_index, mask=mask)
            rmsd_array[idx, idy] = rmsd
            rmsd_array[idy, idx] = rmsd
            tm_array[idx, idy] = tm
            tm_array[idy, idx] = tm
    np.savez_compressed(out_path, rmsd=rmsd_array, tm=tm_array, names=names)

def read_pdb(pdb_path):
    with open(pdb_path) as f:
        pdb_string = f.read()
    result = from_pdb_string(pdb_string)
    return result.atom_positions, result.aatype, result.residue_index, result.chain_index

def parse_slice(slice_str: str) -> slice:
    if slice_str == "none":
        return slice(None)
    result = [0, None, 1]
    position = 0
    while slice_str and position < 3:
        until = slice_str.find(":")
        if until != 0:
            result[position] = int(slice_str[:until])
        slice_str = slice_str[until + 1:]
        position += 1
    return slice(*result)

def parse_options(description, **kwargs):
    parser = ArgumentParser(description=description)
    for kwarg, default in kwargs.items():
        parser.add_argument(f"--{kwarg.replace('_', '-')}", required=False, type=type(default), default=default)
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_options(
        "Evaluate protein design models pairwise RMSD and TM-score.",
        pdb_path="pdb/",
        out_path="out.npz",
    )
    compute_distances(opt.pdb_path, opt.out_path)