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
           complex_index,
           target_structure,
           num_recycles: Optional[int] = 4,
           use_complex_template: bool = True,
           parameter_path: Optional[str] = ""):
    from novobench.alphafold2.model import AlphaFoldUncrop
    model = AlphaFoldUncrop(AF_MODELS['af_1'], parameter_path, num_recycles)
    for name, index, sequence, residue_index, complex_structure in iterator:
        if not use_complex_template:
            complex_structure = None
        scores = model(sequence, residue_index, complex_index, complex_structure, target_structure)
        scores.update(name=name, index=index, sequence=sequence)
        yield scores

def prepare_data(pdb_path: str,
                 target_sequence: str,
                 fasta_path: str,
                 fasta_prefix: Optional[str] = "",
                 fasta_suffix: Optional[str] = "",
                 select_state: Optional[int] = None,
                 section_spec = None,
                 filter_design: Optional[int] = None,
                 homomer: Optional[int] = 1) -> Tuple[str, str, np.ndarray]:
    pdb_files = listdir_nohidden(pdb_path, extensions=("pdb", "pdb1"))
    if select_state is not None:
        tmp_pdb_files = pdb_files
        pdb_files = []
        for name in tmp_pdb_files:
            state = int(name.split(".")[-2])
            if state == select_state:
                pdb_files.append(name)
            pdb_files = sorted(pdb_files)
    if fasta_path == "none":
        for pdb in pdb_files:
            name = ".".join(pdb.split(".")[:-1])
            structure, aatype, residue_index, chain_index = read_pdb(os.path.join(pdb_path, pdb))
            sequence = "".join([restypes[aa] for aa in aatype])
            sequence = chain_sequence(sequence, chain_index)
            sequence = ":".join([target_sequence, sequence.split(":")[1]])
            residue_index = np.arange(sum(len(seq) for seq in sequences), dtype=np.int32)
            yield name, 0, sequence, residue_index, structure
    else:
        fasta_files = listdir_nohidden(fasta_path, extensions=("fa", "fasta"))
        if select_state is not None:
            assert all([".".join(x.split(".")[:-1]) == ".".join(y.split(".")[:-2]) for x, y in zip(fasta_files, pdb_files)])
        else:
            assert all([".".join(x.split(".")[:-1]) == ".".join(y.split(".")[:-1]) for x, y in zip(fasta_files, pdb_files)])
        for pdb, fasta in zip(pdb_files, fasta_files):
            name = ".".join(pdb.split(".")[:-1])
            pdb = os.path.join(pdb_path, pdb)
            fasta = os.path.join(fasta_path, fasta)
            structure, _, residue_index, chain_index = read_pdb(pdb)
            sequences = read_fasta(fasta)
            for index, sequence in enumerate(sequences):
                if filter_design is not None and index != filter_design:
                    continue
                sequence = sequence.replace(":", "")
                if section_spec is not None:
                    select_state = select_state if select_state is not None else 0
                    sequence = expand_sequence(sequence, section_spec)[select_state]
                else:
                    sequence = fasta_prefix + sequence * homomer + fasta_suffix
                    sequence = chain_sequence(sequence, chain_index)
                sequence = ":".join([target_sequence, sequence.split(":")[1]])
                residue_index = np.arange(len(sequence.replace(":", "")), dtype=np.int32)
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
    with open(f"{path}/scores.csv", "w") as f:
        f.write("name,index,sequence,plddt,ptm,pae,ipae,mpae\n")
        for item in scores:
            os.makedirs(f"{path}/predictions/{item['name']}/", exist_ok=True)
            f.write(f"{item['name']},{item['index']},{item['sequence']},{item['plddt']},{item['ptm']},{item['pae']},{item['ipae']},{item['mpae']}\n")
            f.flush()
            write_pdb(f"{path}/predictions/{item['name']}/design_{item['index']}.pdb", item['output'])

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

if __name__ == "__main__":
    opt = parse_options(
        "Uncrop a binder design which was designed on a crop of a target protein (experimental).",
        pdb_path="pdb/",
        fasta_path="fasta/",
        target_structure="target.pdb",
        target_sequence="target.pdb",
        use_complex_template="True",
        section_spec="none",
        select_state="none",
        complex_string="",
        parameter_path="parameters/",
        fasta_prefix="",
        fasta_suffix="",
        homomer=1,
        slice_pdb="none",
        out_path="out.csv",
        model_type="esm",
        num_recycles=4,
        prediction_mode="abinitio",
        templated="none",
        filter_design="none",
    )
    use_complex_template = opt.use_complex_template == "True"
    select_state = None if opt.select_state == "none" else int(opt.select_state)
    filter_design = None if opt.filter_design == "none" else int(opt.filter_design)
    target_structure, _, _, _ = read_pdb(opt.target_structure)
    complex_index = np.array([idx for idx, pos in enumerate(opt.complex_string) if pos == "F"], dtype=np.int32)
    write_result_stream(opt.out_path,
                        runner(prepare_data(opt.pdb_path,
                                            opt.target_sequence,
                                            opt.fasta_path,
                                            select_state=select_state,
                                            section_spec=parse_section_spec(opt.section_spec),
                                            filter_design=filter_design),
                               complex_index,
                               target_structure,
                               use_complex_template=use_complex_template,
                               parameter_path=opt.parameter_path,
                               num_recycles=opt.num_recycles))