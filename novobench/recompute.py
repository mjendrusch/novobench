from argparse import ArgumentParser
from collections import defaultdict
import os

from typing import List, Optional, Tuple

import numpy as np

from alphafold.common.residue_constants import restypes
from alphafold.common.protein import from_pdb_string, to_pdb, Protein

from novobench.analysis.alignment import compute_scores_permuted, compute_scores
from pydssp.pydssp_numpy import assign as dssp_assign

INITIAL_GUESS = defaultdict(default_factory=lambda x: False, initial_guess=True)
AF_MODELS = {
    f"af_{i}{mm}": f"model_{i}{mm}_ptm"
    for i in range(1, 6)
    for mm in ["", "_multimer"]
}
MODEL_TYPES = dict(esm="esmfold_v1", **AF_MODELS)

def runner(iterator,
           #names: List[str],
           #sequences: List[str],
           #structures: List[np.ndarray[np.float32]],
           model_type: Optional[str] = "esm",
           prediction_mode: Optional[str] = "abinitio",
           num_recycles: Optional[int] = 4):
    if model_type == "esm":
        from novobench.esmfold.model import ESMScore
        model = ESMScore()
    elif model_type in AF_MODELS:
        from novobench.alphafold2.model import AlphaFoldScore
        model = AlphaFoldScore(parameter_set=AF_MODELS[model_type])
    else:
        raise ValueError(f"Invalid model type. Given {model_type} but expected one of {MODEL_TYPES}.")
    for name, index, sequence, structure in iterator:
        scores = model(sequence, structure,
                       initial_guess=prediction_mode == "guess",
                       template=prediction_mode == "template",
                       num_recycles=num_recycles)
        scores.update(name=name, index=index, sequence=sequence)
        yield scores

def prepare_data(pdb_path: str,
                 out_path: str,
                 slice_pdb: Optional[str] = "none") -> Tuple[str, str, np.ndarray]:
    slice_pdb = parse_slice(slice_pdb)
    pdb_files = sorted(os.listdir(pdb_path))
    for pdb in pdb_files:
        name = ".".join(pdb.split(".")[:-1])
        structure, aatype, _, chain_index = read_pdb(os.path.join(pdb_path, pdb))
        for prediction_path in sorted(os.listdir(f"{out_path}/predictions/{name}")):
            index = int(prediction_path.split(".")[0].split("_")[-1])
            predicted, aatype, _, _ = read_pdb(os.path.join(out_path, "predictions", name, prediction_path))
            sequence = "".join([restypes[aa] for aa in aatype])
            sequence = chain_sequence(sequence, chain_index)
            structured = (dssp_assign(predicted[:, :4])[..., :2] > 0).any(axis=-1)
            mask = structured
            # rmsd, tm = compute_scores(predicted[:, 1], structure[:, 1], mask=mask)
            rmsd, tm = compute_scores_permuted(predicted[:, 1], structure[:, 1], chain_index, mask=mask)
            yield dict(name=name, index=index, sequence=sequence, sc_rmsd=rmsd, sc_tm=tm)

def chain_sequence(sequence, chain_index):
    result = []
    current_idx = chain_index[0]
    for aa, idx in zip(sequence, chain_index):
        if idx != current_idx:
            current_idx = idx
            result.append(":")
        result.append(aa)
    return "".join(result)

def read_pdb(pdb_path):
    with open(pdb_path) as f:
        pdb_string = f.read()
    result = from_pdb_string(pdb_string)
    return result.atom_positions, result.aatype, result.residue_index, result.chain_index

def read_fasta(fasta_path):
    with open(fasta_path) as f:
        while True:
            try:
                _ = next(f)
                sequence = next(f).strip()
                yield sequence
            except StopIteration:
                break

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

def write_result_stream(path, scores):
    os.makedirs(f"{path}/predictions/", exist_ok=True)
    with open(f"{path}/scores_recomputed.csv", "w") as f:
        f.write("name,index,sequence,sc_rmsd,sc_tm\n")
        for item in scores:
            f.write(f"{item['name']},{item['index']},{item['sequence']},{item['sc_rmsd']},{item['sc_tm']}\n")

def write_pdb(path, outputs):
    protein = Protein(
        outputs['positions'],
        outputs['aatype'][0],
        outputs['atom37_atom_exists'][0],
        outputs['residue_index'][0],
        np.zeros_like(outputs['residue_index'][0]),
        outputs['plddt'][0]
    )
    pdb_string = to_pdb(protein)
    with open(path, "w") as f:
        f.write(pdb_string)

def parse_options(description, **kwargs):
    parser = ArgumentParser(description=description)
    for kwarg, default in kwargs.items():
        parser.add_argument(f"--{kwarg.replace('_', '-')}", required=False, type=type(default), default=default)
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_options(
        "Evaluate protein design models with self-consistence under structure prediction.",
        pdb_path="pdb/",
        out_path="out.csv",
    )
    write_result_stream(opt.out_path,
                        prepare_data(opt.pdb_path, opt.out_path))