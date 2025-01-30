# copyright (c) 2023 Michael Jendrusch, EMBL
# AlphaDesign: A de novo protein design framework based on AlphaFold
# Michael Jendrusch, Alessio Ling Jie Yang, Elisabetta Cacace, Jacob Bobonis,
# Carlos Geert Pieter Voogdt, Athanasios Typas, Jan O. Korbel, and S. Kashif Sadiq

from argparse import ArgumentParser
from collections import defaultdict
import os

from typing import List, Optional, Tuple

import numpy as np

from alphafold.common.residue_constants import restypes
from alphafold.common.protein import from_pdb_string, to_pdb, Protein

from novobench.utils import listdir_nohidden

INITIAL_GUESS = defaultdict(default_factory=lambda x: False, initial_guess=True)
AF_MODELS = {
    f"af_{i}{mm}": f"model_{i}{mm}_ptm"
    for i in range(1, 6)
    for mm in ["", "_multimer"]
}
MODEL_TYPES = dict(esm="esmfold_v1", **AF_MODELS)

def runner(iterator,
           model_type: Optional[str] = "esm",
           prediction_mode: Optional[str] = "abinitio",
           templated: Optional[List[int]] = None,
           num_recycles: Optional[int] = 4,
           parameter_path: Optional[str] = ""):
    if model_type == "esm":
        from novobench.esmfold.model import ESMScore
        model = ESMScore()
    elif model_type in AF_MODELS:
        from novobench.alphafold2.model import AlphaFoldScore
        model = AlphaFoldScore(AF_MODELS[model_type], parameter_path, num_recycles)
    else:
        raise ValueError(f"Invalid model type. Given {model_type} but expected one of {MODEL_TYPES}.")
    for name, index, sequence, residue_index, structure in iterator:
        scores = model(sequence, structure, residue_index,
                       initial_guess=prediction_mode == "guess",
                       templated=templated,
                       num_recycles=num_recycles)
        scores.update(name=name, index=index, sequence=sequence)
        yield scores

def prepare_data(pdb: str,
                 fasta: str,
                 fasta_prefix: Optional[str] = "",
                 fasta_suffix: Optional[str] = "",
                 homomer: Optional[int] = 1,
                 slice_pdb: Optional[str] = "none") -> Tuple[str, str, np.ndarray]:
    slice_pdb = parse_slice(slice_pdb)
    name = ".".join(pdb.split(".")[:-1])
    structure, _, residue_index, chain_index = read_pdb(pdb)
    sequences = read_alphalog(fasta)
    for index, sequence in enumerate(sequences):
        sequence = sequence.replace(":", "")
        sequence = fasta_prefix + sequence * homomer + fasta_suffix
        sequence = chain_sequence(sequence, chain_index)
        yield name, index, sequence, residue_index, structure

def expand_sequence(sequence, section_spec):
    return [
        ":".join([sequence[span] for span in state])
        for state in section_spec
    ]

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

def read_alphalog(path):
    with open(path) as f:
        for index, line in enumerate(f):
            sequence, score, _ = line.strip().split(",")
            yield sequence.strip()

def read_alphalog_scores(path):
    with open(path) as f:
        for index, line in enumerate(f):
            sequence, score, _ = line.strip().split(",")
            yield float(score)

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

def write_result_stream(path, scores, logfile):
    os.makedirs(f"{path}/steps/", exist_ok=True)
    with open(f"{path}/scores.csv", "w") as f:
        f.write("iteration,rank,fitness,plddt,ptm,pae,ipae,mpae\n")
        for idx, (item, fitness) in enumerate(zip(scores, read_alphalog_scores(logfile))):
            iteration = idx // 10
            rank = idx % 10
            f.write(f"{iteration},{rank},{fitness},{item['plddt']},{item['ptm']},{item['pae']},{item['ipae']},{item['mpae']}\n")
            f.flush()
            write_pdb(f"{path}/steps/snapshot_{iteration}_{rank}.pdb", item['output'])

def write_pdb(path, outputs):
    protein = Protein(
        outputs['positions'],
        outputs['aatype'][0],
        outputs['atom37_atom_exists'][0],
        outputs['residue_index'][0],
        outputs['chain_index'][0],
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

def parse_templated(data):
    if data == "none":
        return None
    return [int(c) for c in data.strip().split(",")]

def parse_section_spec(data):
    if data == "none":
        return None
    def pseudo_int(x: str) -> Optional[int]:
        if x == "None":
            return None
        return int(x)
    spans = [[slice(*map(pseudo_int, span.split(":"))) for span in spec.split(",")] for spec in data.strip().split(";")]
    return spans

def run_colab(out_path, pdb_path, fasta_path, parameter_path, num_recycles=4):
    write_result_stream(out_path,
                        runner(prepare_data(pdb_path, fasta_path,
                                            fasta_prefix="",
                                            fasta_suffix="",
                                            homomer=1),
                               model_type="af_1",
                               parameter_path=parameter_path,
                               prediction_mode="guess",
                               templated=parse_templated("none"),
                               num_recycles=num_recycles))

if __name__ == "__main__":
    opt = parse_options(
        "Evaluate protein design models with self-consistence under structure prediction.",
        pdb_path="pseudo.pdb",
        log_path="full_log",
        parameter_path="parameters/",
        fasta_prefix="",
        fasta_suffix="",
        homomer=1,
        select_state="none",
        section_spec="none",
        slice_pdb="none",
        out_path="out.csv",
        model_type="esm",
        num_recycles=4,
        prediction_mode="abinitio",
        templated="none"
    )
    write_result_stream(opt.out_path,
                        runner(prepare_data(opt.pdb_path, opt.log_path,
                                            fasta_prefix=opt.fasta_prefix,
                                            fasta_suffix=opt.fasta_suffix,
                                            homomer=opt.homomer),
                               model_type=opt.model_type,
                               parameter_path=opt.parameter_path,
                               prediction_mode=opt.prediction_mode,
                               templated=parse_templated(opt.templated),
                               num_recycles=opt.num_recycles),
                        opt.log_path)