# copyright (c) 2023 Michael Jendrusch, EMBL
# AlphaDesign: A de novo protein design framework based on AlphaFold
# Michael Jendrusch, Alessio Ling Jie Yang, Elisabetta Cacace, Jacob Bobonis,
# Carlos Geert Pieter Voogdt, Athanasios Typas, Jan O. Korbel, and S. Kashif Sadiq

import numpy as np
import esm
from novobench.analysis.alignment import compute_scores_permuted
from openfold.utils.feats import atom14_to_atom37
from pydssp.pydssp_numpy import assign as dssp_assign

class ESMScore:
    def __init__(self):
        self.model = esm.pretrained.esmfold_v1()
        self.model.eval()
        self.model.cuda()

    def __call__(self, sequence, structure, residue_index, initial_guess=False, templated=None, mask_loops=True, num_recycles=4):
        output = self.model.infer(sequence, num_recycles=num_recycles, chain_linker="")
        predicted = atom14_to_atom37(output["positions"][-1], output).to("cpu").numpy()
        predicted = predicted[0]
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        output["positions"] = predicted
        mask = None
        if mask_loops:
            structured = (dssp_assign(predicted[:, :4])[..., :2] > 0).any(axis=-1)
            mask = structured
        plddt = output["mean_plddt"][0]
        ptm = output["ptm"][0]
        pae = output["predicted_aligned_error"][0]
        chain_index = output["chain_index"][0]
        same_chain = chain_index[:, None] == chain_index[None, :]
        ipae = np.where(same_chain, 0, pae).sum() / (1 - same_chain).sum()
        mpae = np.where(same_chain, 10_000, pae).min()
        pae = pae.mean()
        predicted = predicted[:, 1]
        structure = structure[:, 1]
        rmsd, tm = compute_scores_permuted(predicted, structure, chain_index, mask)
        return dict(plddt=plddt, ptm=ptm, pae=pae, ipae=ipae, mpae=mpae, sc_rmsd=rmsd, sc_tm=tm, output=output)
