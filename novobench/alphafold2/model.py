# copyright (c) 2023 Michael Jendrusch, EMBL
# AlphaDesign: A de novo protein design framework based on AlphaFold
# Michael Jendrusch, Alessio Ling Jie Yang, Elisabetta Cacace, Jacob Bobonis,
# Carlos Geert Pieter Voogdt, Athanasios Typas, Jan O. Korbel, and S. Kashif Sadiq

import jax
import jax.numpy as jnp

# custom alphafold
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from alphafold.model import config as afconfig
from alphafold.model import data as afdata
from alphafold.common import residue_constants
from alphafold.model.modules import AlphaFoldIteration
# from alphafold.model.modules import pseudo_beta_fn
from alphafold.model.prng import SafeKey
from alphafold.model.utils import mask_mean

from pydssp.pydssp_numpy import assign as dssp_assign
from novobench.analysis.alignment import compute_scores_permuted

hk.vmap.require_split_rng = False

def compute_pseudo_cb(positions):
    n, ca, co = jnp.moveaxis(positions[..., :3, :], -2, 0)
    b = ca - n
    c = co - ca
    a = jnp.cross(b, c)
    const = [-0.58273431, 0.56802827, -0.54067466]
    return const[0] * a + const[1] * b + const[2] * c + ca

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = jnp.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    has_cb = all_atom_masks[..., cb_idx]
    ca_pos = all_atom_positions[..., ca_idx, :]
    cb_pos = all_atom_positions[..., cb_idx, :]
    cb_pos = jnp.where(
        has_cb[..., None],
        cb_pos,
        compute_pseudo_cb(all_atom_positions)
    )
    pseudo_beta = jnp.where(
        jnp.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        ca_pos, cb_pos)
    return pseudo_beta

class AlphaFoldModel(hk.Module):
    def __init__(self, config, name="alphafold"):
        super().__init__(name=name)
        self.config = config
        self.global_config = config.global_config

    def __call__(self, batch):
        """Runs AlphaFold on a batch of input data.
        
        Args:
            batch (Dict[str, ndarray]): Dictionary of input data to AlphaFold. 

        Returns:
            A dictionary of output values containing the
            predicted structure information (["structure"]["all_atom_positions"]),
            as well as confidence values.
        """
        impl = AlphaFoldIteration(self.config, self.global_config)
        batch_size, num_residues = batch['aatype'].shape

        # get data for recycling from the previous
        # iteration of AlphaFold. This includes
        # the previous position "prev_pos",
        # the previous MSA embedding for the
        # input sequence "prev_msa_first_row",
        # and the previous pair embedding "prev_pair"
        def get_prev(ret):
            new_prev = {
                'prev_pos':
                    ret['structure_module']['final_atom_positions'],
                'prev_msa_first_row': ret['representations']['msa_first_row'],
                'prev_pair': ret['representations']['pair'],
            }
            return jax.tree_map(jax.lax.stop_gradient, new_prev)

        # run a single iteration of AlphaFold with recycling
        def do_call(prev,
                    recycle_idx):
            if self.config.resample_msa_in_recycling:
                num_ensemble = batch_size // (self.config.num_recycle + 1)
                def slice_recycle_idx(x):
                    start = recycle_idx * num_ensemble
                    size = num_ensemble
                    return jax.lax.dynamic_slice_in_dim(x, start, size, axis=0)
                ensembled_batch = jax.tree_map(slice_recycle_idx, batch)
            else:
                num_ensemble = batch_size
                ensembled_batch = batch
            non_ensembled_batch = jax.tree_map(lambda x: x, prev)

            return impl(
                ensembled_batch=ensembled_batch,
                non_ensembled_batch=non_ensembled_batch,
                is_training=False,
                compute_loss=False,
                ensemble_representations=False)

        # initialise recycling data (prev)
        prev = {}
        emb_config = self.config.embeddings_and_evoformer
        if emb_config.recycle_pos:
            prev['prev_pos'] = jnp.zeros(
                [num_residues, residue_constants.atom_type_num, 3])
            # if we provide an initial guess of the structure,
            # initialise the recycling position input "prev_pos"
            # with that guess
            if 'initial_guess' in batch:
                prev['prev_pos'] = batch['initial_guess']
        if emb_config.recycle_features:
            prev['prev_msa_first_row'] = jnp.zeros(
                [num_residues, emb_config.msa_channel])
            prev['prev_pair'] = jnp.zeros(
                [num_residues, num_residues, emb_config.pair_channel])

        # run num_recycle iterations of AlphaFold
        # on the input sequence
        num_iter = self.config.num_recycle
        body = lambda x: (x[0] + 1,  # pylint: disable=g-long-lambda
                          get_prev(do_call(x[1], recycle_idx=x[0])))
        _, prev = hk.while_loop(
            lambda x: x[0] < num_iter,
            body,
            (0, prev))

        ret = do_call(prev=prev, recycle_idx=num_iter)
        return ret

class AlphaFoldUncrop:
    def __init__(self, model, path, num_recycles=4) -> None:
        config = afconfig.model_config(model)
        config.model.resample_msa_in_recycling = False
        config.model.num_recycle = num_recycles - 1
        config.model.embeddings_and_evoformer.template.enabled = True
        config.data.common.max_extra_msa = 1
        config.data.eval.max_msa_clusters = 5
        config.data.eval.max_templates = 2
        self.config = config
        params = afdata.get_model_haiku_params(model_name=model, data_dir=path)
        module_apply = hk.transform(lambda x: AlphaFoldModel(self.config.model, name="alphafold")(x)).apply
        self.module = jax.jit(lambda x: module_apply(params, jax.random.PRNGKey(42), x))

    def __call__(self, sequence, residue_index, complex_index, complex_structure, target_structure):
        features = make_uncrop_features(sequence, complex_index, complex_structure, target_structure)
        offset = [0]
        lengths = []
        for seq in sequence.split(":"):
            offset.append(offset[-1] + len(seq) + 200)
            lengths.append(len(seq))
        offset = jnp.repeat(jnp.array(offset[:-1]), jnp.array(lengths))
        features["residue_index"] = residue_index[None] + offset[None]
        features = jax.tree_map(lambda x: x, features)
        results = self.module(features)
        plddt = self.get_plddt(results)
        ptm = self.get_ptm(results)
        pae = self.get_pae(results)
        same_chain = features["chain_index"][0, :, None] == features["chain_index"][0, None, :]
        ipae = jnp.where(same_chain, 0, pae).sum() / (1 - same_chain).sum()
        mpae = jnp.where(same_chain, jnp.inf, pae).min()
        pae = pae.mean()
        all_atom_positions = results["structure_module"]["final_atom_positions"]
        output = dict(
            residue_index=features["residue_index"],
            chain_index=features["chain_index"],
            positions=all_atom_positions,
            atom_mask=features["atom37_atom_exists"],
            aatype=features["aatype"],
            plddt=jnp.repeat(plddt[..., None], 37, axis=-1),
            atom37_atom_exists=features["atom37_atom_exists"]
        )
        output = {
            name: np.array(output[name])
            for name in output
        }
        mask = None
        structured = (dssp_assign(all_atom_positions[:, :4])[..., :2] > 0).any(axis=-1)
        mask = structured
        plddt = plddt[0].mean()
        return dict(plddt=plddt, ptm=ptm, pae=pae, ipae=ipae, mpae=mpae, output=output)

    def get_plddt(self, data):
        lddt_logits = data["predicted_lddt"]["logits"]
        bin_centers = jnp.arange(0, lddt_logits.shape[-1]) + 1 / 2
        bin_centers = bin_centers / lddt_logits.shape[-1]
        lddt = jax.nn.softmax(lddt_logits, axis=-1) * bin_centers[None, None, :]
        lddt = lddt.sum(axis=-1)
        return lddt * 100.0

    def get_pae(self, data):
        logits = data["predicted_aligned_error"]["logits"]
        breaks = data["predicted_aligned_error"]["breaks"]
        bin_centers = _calculate_bin_centers(breaks)
        pae = jax.nn.softmax(logits, axis=-1) * bin_centers[None, None, :]
        pae = pae.sum(axis=-1)
        return pae

    def get_ptm(self, data):
        return predicted_tm_score(
            data["predicted_aligned_error"]["logits"],
            data["predicted_aligned_error"]["breaks"],
        )

class AlphaFoldScore:
    def __init__(self, model, path, num_recycles=4, max_templates=1, name: Optional[str] = "af_score"):
        config = afconfig.model_config(model)
        config.model.resample_msa_in_recycling = False
        config.model.num_recycle = num_recycles - 1
        config.model.embeddings_and_evoformer.template.enabled = True
        config.data.common.max_extra_msa = 1
        config.data.eval.max_msa_clusters = 5
        config.data.eval.max_templates = max_templates
        self.config = config
        params = afdata.get_model_haiku_params(model_name=model, data_dir=path)
        # params = {f"alphafold/{key}": value for key, value in params.items()}
        module_apply = hk.transform(lambda x: AlphaFoldModel(self.config.model, name="alphafold")(x)).apply
        self.module = jax.jit(lambda x: module_apply(params, jax.random.PRNGKey(42), x))

    def __call__(self, sequence, structure, residue_index, initial_guess=False, templated=None, mask_loops=True, num_recycles=4):
        features = make_af_features(sequence, structure, templated=templated)
        if initial_guess:
            features["initial_guess"] = structure#[None, None]
        offset = [0]
        lengths = []
        for seq in sequence.split(":"):
            offset.append(offset[-1] + len(seq) + 200)
            lengths.append(len(seq))
        offset = jnp.repeat(jnp.array(offset[:-1]), jnp.array(lengths))
        features["residue_index"] = residue_index[None] + offset[None]
        templated_mask = jnp.ones_like(features["aatype"])[0]
        if templated is not None:
            templated_mask = (features["chain_index"][0][..., None] != jnp.array(templated, dtype=jnp.int32)).any(axis=-1)
        features = jax.tree_map(lambda x: x, features)
        results = self.module(features)
        plddt = self.get_plddt(results)
        ptm = self.get_ptm(results)
        pae = self.get_pae(results)
        same_chain = features["chain_index"][0, :, None] == features["chain_index"][0, None, :]
        ipae = jnp.where(same_chain, 0, pae).sum() / (1 - same_chain).sum()
        mpae = jnp.where(same_chain, jnp.inf, pae).min()
        pae = pae.mean()
        all_atom_positions = results["structure_module"]["final_atom_positions"]
        output = dict(
            residue_index=features["residue_index"],
            chain_index=features["chain_index"],
            positions=all_atom_positions,
            atom_mask=features["atom37_atom_exists"],
            aatype=features["aatype"],
            plddt=jnp.repeat(plddt[..., None], 37, axis=-1),
            atom37_atom_exists=features["atom37_atom_exists"]
        )
        output = {
            name: np.array(output[name])
            for name in output
        }
        mask = None
        if mask_loops:
            structured = (dssp_assign(all_atom_positions[:, :4])[..., :2] > 0).any(axis=-1)
            mask = structured
        rmsd, tm = compute_scores_permuted(output["positions"][:, 1], structure[:, 1], output["chain_index"], mask)
        plddt = (plddt[0] * templated_mask).sum() / jnp.maximum(templated_mask.sum(), 1e-6)
        return dict(plddt=plddt, ptm=ptm, pae=pae, ipae=ipae, mpae=mpae, sc_rmsd=rmsd, sc_tm=tm, output=output)

    def get_plddt(self, data):
        lddt_logits = data["predicted_lddt"]["logits"]
        bin_centers = jnp.arange(0, lddt_logits.shape[-1]) + 1 / 2
        bin_centers = bin_centers / lddt_logits.shape[-1]
        lddt = jax.nn.softmax(lddt_logits, axis=-1) * bin_centers[None, None, :]
        lddt = lddt.sum(axis=-1)
        return lddt * 100.0

    def get_pae(self, data):
        logits = data["predicted_aligned_error"]["logits"]
        breaks = data["predicted_aligned_error"]["breaks"]
        bin_centers = _calculate_bin_centers(breaks)
        pae = jax.nn.softmax(logits, axis=-1) * bin_centers[None, None, :]
        pae = pae.sum(axis=-1)
        return pae

    def get_ptm(self, data):
        return predicted_tm_score(
            data["predicted_aligned_error"]["logits"],
            data["predicted_aligned_error"]["breaks"],
        )

def make_af_score(
    model="model_1_ptm", path=None, num_recycle=2, num_chains=1, params=None
):
    config = afconfig.model_config(model)
    config.model.resample_msa_in_recycling = False
    config.model.num_recycle = num_recycle - 1
    config.model.embeddings_and_evoformer.template.enabled = True
    config.data.common.max_extra_msa = 1
    config.data.eval.max_msa_clusters = num_chains
    config.data.eval.max_templates = 1
    if params is None:
        params = afdata.get_model_haiku_params(model_name=model, data_dir=path)
    params = {f"af_score/{key}": value for key, value in params.items()}

    def inner(sequences, templates=None):
        return hk.transform(lambda x: AlphaFoldScore(config.model)(x)).apply(
            params, hk.next_rng_key(), sequences
        )

    return inner


def make_af_multi_score(path, num_recycle=2, num_chains=1):
    config = afconfig.model_config("model_1_ptm")
    config.model.resample_msa_in_recycling = False
    config.model.num_recycle = num_recycle - 1
    config.model.embeddings_and_evoformer.template.enabled = True
    config.data.common.max_extra_msa = 1
    config.data.eval.max_msa_clusters = num_chains
    config.data.eval.max_templates = 1
    param_list = []
    for idx in range(1, 6):
        params = afdata.get_model_haiku_params(
            model_name=f"model_{idx}_ptm", data_dir=path
        )
        params = {f"af_score/{key}": value for key, value in params.items()}
        param_list.append(params)

    def inner(sequences):
        af_model = hk.transform(lambda x: AlphaFoldScore(config.model)(x)).apply
        results = []
        for params in param_list:
            model_n_result = af_model(params, hk.next_rng_key(), sequences)
        results.append(model_n_result)

    return inner


# NOTE: adapted from AlphaFold confidence computation
def predicted_tm_score(
    logits: jnp.ndarray,
    breaks: jnp.ndarray,
    residue_weights: Optional[jnp.ndarray] = None,
    asym_id: Optional[jnp.ndarray] = None,
    interface: bool = False,
) -> jnp.ndarray:
    """Computes predicted TM alignment or predicted interface TM alignment score.

    Args:
        logits: [num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
        breaks: [num_bins] the error bins.
        residue_weights: [num_res] the per residue weights to use for the
        expectation.
        asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
        ipTM calculation, i.e. when interface=True.
        interface: If True, interface predicted TM score is computed.

    Returns:
        ptm_score: The predicted TM alignment or the predicted iTM score.
    """

    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = jnp.ones(logits.shape[0])

    bin_centers = _calculate_bin_centers(breaks)

    num_res = logits.shape[0]
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = num_res  # max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.
    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    pair_mask = jnp.ones(shape=(num_res, num_res), dtype=bool)
    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment[(per_alignment * residue_weights).argmax()]


def _calculate_bin_centers(breaks: jnp.ndarray):
    """Gets the bin centers from the bin edges.

    Args:
      breaks: [num_bins - 1] the error bin edges.

    Returns:
      bin_centers: [num_bins] the error bin centers.
    """
    step = breaks[1] - breaks[0]

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = jnp.concatenate([bin_centers, bin_centers[-1:] + step], axis=0)
    return bin_centers

def make_uncrop_features(sequence, complex_indices, complex_structure, target_structure):
    sequences = [encode_sequence(s)[None] for s in sequence.split(":")]
    template_features = make_uncrop_template_features(sequences, complex_indices, complex_structure, target_structure)
    return {
        "is_distillation": jnp.zeros((1,)),
        "random_crop_to_size_seed": jnp.array(
            [[-2055043226, 1526785730]], dtype=jnp.int32
        ),
        "residue_index": make_residue_index(sequences, offset=200),
        "chain_index": make_chain_index(sequences),
        **make_sequence_feat(sequences),
        **template_features,
        **make_msa_features(sequences),
        **make_atom14_masks(sequences),
    }

def make_af_features(sequence, structure, offset=200, templated=None):
    sequences = [encode_sequence(s)[None] for s in sequence.split(":")]
    return {
        "is_distillation": jnp.zeros((1,)),
        "random_crop_to_size_seed": jnp.array(
            [[-2055043226, 1526785730]], dtype=jnp.int32
        ),
        "residue_index": make_residue_index(sequences, offset=offset),
        "chain_index": make_chain_index(sequences),
        **make_sequence_feat(sequences),
        **make_template_features(sequences, structure, templated=templated),
        **make_msa_features(sequences),
        **make_atom14_masks(sequences),
    }

def encode_sequence(sequence):
    return jax.nn.one_hot(jnp.array([residue_constants.restypes.index(c) if c in residue_constants.restypes else 0 for c in sequence]), 20, axis=-1)

def make_sequence_feat(sequences):
    raw_sequence = jnp.concatenate(sequences, axis=1)
    sequence = jnp.concatenate(
        [
            jnp.zeros((1, raw_sequence.shape[1], 1)),  # extend by domain break
            raw_sequence,
            jnp.zeros((1, raw_sequence.shape[1], 1)),  # extend by deletion
        ],
        axis=2,
    )
    return dict(
        aatype=jnp.argmax(raw_sequence, axis=2),
        target_feat=sequence,
        seq_mask=jnp.ones((1, sequence.shape[1])),
        seq_length=sequence.shape[1] * jnp.ones((1,), dtype=jnp.int32),
    )

def make_residue_index(sequences, offset=200):
    results = []
    current_offset = 0
    for sequence in sequences:
        results.append(jnp.arange(0, sequence.shape[1], dtype=jnp.int32) + current_offset)
        current_offset += sequence.shape[1] + offset
    return jnp.concatenate(results, axis=0)[None, :]

def make_chain_index(sequences):
    results = []
    count = 0
    for sequence in sequences:
        results.append(jnp.ones(sequence.shape[1], dtype=jnp.int32) * count)
        count += 1
    return jnp.concatenate(results, axis=0)[None, :]

def make_seq_len(sequences):
    return sum(sequence.shape[1] for sequence in sequences)

def make_template_features(sequences, structure, templated=None):
    size = make_seq_len(sequences)
    sequences = [jnp.argmax(s, axis=-1) for s in sequences]
    if not templated:
        aatype = jnp.zeros((1, 1, size), dtype=jnp.int32)
        # aatype = aatype.at[:, 0].set(21)
        # aatype[:, 0] = 21
        template_mask = jnp.zeros((1, 1))
        # template_mask = template_mask.at[:, 0].set(1)
        # template_mask[:, 0] = 1
        template_pseudo_beta = jnp.zeros((1, 1, size, 3))
        template_pseudo_beta_mask = jnp.zeros((1, 1, size))

        template_all_atom_positions = jnp.zeros((1, 1, size, 37, 3))
        template_all_atom_mask = jnp.zeros((1, 1, size, 37))
    else:
        aatype = []
        template_all_atom_positions = []
        template_all_atom_mask = []
        template_pseudo_beta = []
        template_pseudo_beta_mask = []
        templated_mask = jnp.concatenate([
            jnp.ones_like(seq, dtype=jnp.bool_)
            if idx in templated
            else jnp.zeros_like(seq, dtype=jnp.bool_)
            for idx, seq in enumerate(sequences)
        ], axis=1)
        aatype = jnp.concatenate([
            seq
            if idx in templated
            else jnp.zeros_like(seq, dtype=jnp.bool_)
            for idx, seq in enumerate(sequences)            
        ], axis=1)[None]
        all_sequence = jnp.concatenate(sequences, axis=1)
        all_atom, all_atom_mask = (
            structure,
            make_atom14_masks([jax.nn.one_hot(s, 20, axis=-1) for s in sequences])["atom37_atom_exists"],
        )
        # NOTE: when the template structure does not contain Cb atoms,
        # add Cb positions retroactively. Otherwise AlphaFold is unable
        # to properly use the template structure
        all_atom = jnp.array(all_atom)
        all_atom = all_atom.at[..., residue_constants.atom_order["CB"], :].set(compute_pseudo_cb(all_atom))
        all_atom_mask = all_atom_mask.at[..., residue_constants.atom_order["CB"]].set(1)
        all_atom_mask *= templated_mask[:, :, None]
        cb_coordinates = pseudo_beta_fn(aatype, all_atom, all_atom_mask)
        template_all_atom_positions.append(all_atom[None, None])
        template_all_atom_mask.append(all_atom_mask[None])
        template_pseudo_beta.append(cb_coordinates)
        cb_mask = jnp.any(all_atom_mask, axis=-1)
        template_pseudo_beta_mask.append(cb_mask[None])
        template_all_atom_positions = jnp.concatenate(
            template_all_atom_positions, axis=1
        )
        template_all_atom_mask = jnp.concatenate(template_all_atom_mask, axis=1)
        template_pseudo_beta = jnp.concatenate(template_pseudo_beta, axis=1)
        template_pseudo_beta_mask = jnp.concatenate(template_pseudo_beta_mask, axis=1)
        template_mask = jnp.ones((1, 1))
    result = dict(
        template_all_atom_positions=template_all_atom_positions,
        template_all_atom_masks=template_all_atom_mask,
        template_aatype=aatype,
        template_mask=template_mask,
        template_pseudo_beta=template_pseudo_beta,
        template_pseudo_beta_mask=template_pseudo_beta_mask,
    )

    return result

def make_uncrop_template_features(sequences, complex_indices, complex_structure, target_structure):
    sequences = [jnp.argmax(s, axis=-1) for s in sequences]
    aatype = []
    template_all_atom_positions = []
    template_all_atom_mask = []
    template_pseudo_beta = []
    template_pseudo_beta_mask = []
    aatype = jnp.concatenate(sequences, axis=1)

    # make complex template
    complex_template = None
    if complex_structure is not None:
        all_atom, all_atom_mask = (
            complex_structure,
            make_atom14_masks([jax.nn.one_hot(s, 20, axis=-1) for s in sequences])["atom37_atom_exists"],
        )
        all_atom = jnp.zeros((aatype.shape[1], 37, 3), dtype=jnp.float32).at[complex_indices].set(all_atom)
        all_atom_mask = jnp.zeros((aatype.shape[1], 37), dtype=jnp.float32).at[complex_indices].set(all_atom_mask[0, complex_indices])[None]
        cb_coordinates = pseudo_beta_fn(aatype, all_atom, None)
        template_all_atom_positions.append(all_atom[None, None])
        template_all_atom_mask.append(all_atom_mask[None])
        template_pseudo_beta.append(cb_coordinates[None])
        cb_mask = jnp.any(all_atom_mask, axis=-1)
        template_pseudo_beta_mask.append(cb_mask[None])
        template_all_atom_positions = jnp.concatenate(
            template_all_atom_positions, axis=1
        )
        template_all_atom_mask = jnp.concatenate(template_all_atom_mask, axis=1)
        template_pseudo_beta = jnp.concatenate(template_pseudo_beta, axis=1)
        template_pseudo_beta_mask = jnp.concatenate(template_pseudo_beta_mask, axis=1)
        template_mask = jnp.ones((1, 1))

        complex_template = dict(
            template_all_atom_positions=template_all_atom_positions,
            template_all_atom_masks=template_all_atom_mask,
            template_aatype=aatype[None],
            template_mask=template_mask,
            template_pseudo_beta=template_pseudo_beta,
            template_pseudo_beta_mask=template_pseudo_beta_mask,
        )

    # make target-only template
    template_all_atom_positions = []
    template_all_atom_mask = []
    template_pseudo_beta = []
    template_pseudo_beta_mask = []
    all_atom, all_atom_mask = (
        target_structure,
        make_atom14_masks([jax.nn.one_hot(s, 20, axis=-1) for s in sequences])["atom37_atom_exists"],
    )
    all_atom = jnp.zeros((aatype.shape[1], 37, 3), dtype=jnp.float32).at[:sequences[0].shape[1]].set(all_atom)
    all_atom_mask = jnp.zeros((aatype.shape[1], 37), dtype=jnp.float32).at[:sequences[0].shape[1]].set(all_atom_mask[0, :sequences[0].shape[1]])[None]
    cb_coordinates = pseudo_beta_fn(aatype, all_atom, None)
    template_all_atom_positions.append(all_atom[None, None])
    template_all_atom_mask.append(all_atom_mask[None])
    template_pseudo_beta.append(cb_coordinates[None])
    cb_mask = jnp.any(all_atom_mask, axis=-1)
    template_pseudo_beta_mask.append(cb_mask[None])
    template_all_atom_positions = jnp.concatenate(
        template_all_atom_positions, axis=1
    )
    template_all_atom_mask = jnp.concatenate(template_all_atom_mask, axis=1)
    template_pseudo_beta = jnp.concatenate(template_pseudo_beta, axis=1)
    template_pseudo_beta_mask = jnp.concatenate(template_pseudo_beta_mask, axis=1)
    template_mask = jnp.ones((1, 1))

    target_template = dict(
        template_all_atom_positions=template_all_atom_positions,
        template_all_atom_masks=template_all_atom_mask,
        template_aatype=aatype[None],
        template_mask=template_mask,
        template_pseudo_beta=template_pseudo_beta,
        template_pseudo_beta_mask=template_pseudo_beta_mask,
    )

    templates = [t for t in [complex_template, target_template] if t is not None]

    result = target_template
    if complex_template is not None:
        result = jax.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), *templates)

    return result

def make_msa_features(sequences):
    seq_lengths = [sequence.shape[1] for sequence in sequences]
    total = sum(seq_lengths)
    results = []
    for idx, sequence in enumerate(sequences):
        unknown = jnp.zeros((1, total, 1))
        bert = unknown
        msa_gap = jnp.zeros((1, total, 1))
        msa_gap = msa_gap.at[: sum(seq_lengths[:idx])].set(1)
        # msa_gap[:sum(seq_lengths[:idx])] = 1
        msa_gap = msa_gap.at[sum(seq_lengths[: idx + 1]) :].set(1)
        # msa_gap[sum(seq_lengths[:idx + 1]):] = 1
        deletion_data = jnp.zeros((1, total, 3))
        padded_sequence = jnp.concatenate(
            [
                jnp.zeros((1, sum(seq_lengths[:idx]), 20)),
                sequence,
                jnp.zeros((1, sum(seq_lengths[idx + 1 :]), 20)),
            ],
            axis=1,
        )
        result = jnp.concatenate(
            [
                padded_sequence,
                unknown,
                msa_gap,
                bert,
                # subsumes has_deletion, deletion_value, deletion_mean_value:
                deletion_data,
                # cluster profile:
                padded_sequence,
                unknown,
                msa_gap,
                bert,
            ],
            axis=2,
        )
        results.append(result[:, None, :, :])
    msa_feat = jnp.concatenate(results, axis=1)
    return dict(
        msa_feat=msa_feat,
        msa_mask=jnp.ones((1, len(sequences), total)),
        bert_mask=jnp.zeros((1, len(sequences), total)),
        msa_row_mask=jnp.ones((1, len(sequences))),
        true_msa=jnp.argmax(msa_feat, axis=-1),
        extra_msa=jnp.zeros((1, 1, total)),
        extra_msa_mask=jnp.zeros((1, 1, total)),
        extra_msa_row_mask=jnp.zeros((1, 1)),
        extra_has_deletion=jnp.zeros((1, 1, total)),
        extra_deletion_value=jnp.zeros((1, 1, total)),
    )


def make_atom14_masks(sequences):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    # Adapted from main Alphafold code to work without Tensorflow
    aatype = jnp.argmax(jnp.concatenate(sequences, axis=1), axis=2)

    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]
        ]

        restype_atom14_to_atom37.append(
            [(residue_constants.atom_order[name] if name else 0) for name in atom_names]
        )

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in residue_constants.atom_types
            ]
        )

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = jnp.array(restype_atom14_to_atom37, dtype=jnp.int32)
    restype_atom37_to_atom14 = jnp.array(restype_atom37_to_atom14, dtype=jnp.int32)
    restype_atom14_mask = jnp.array(restype_atom14_mask, dtype=jnp.float32)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    residx_atom14_mask = restype_atom14_mask[aatype]

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[aatype]

    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], dtype=jnp.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    restype_atom37_mask = jnp.asarray(restype_atom37_mask)
    residx_atom37_mask = restype_atom37_mask[aatype]

    return dict(
        atom14_atom_exists=residx_atom14_mask,
        residx_atom14_to_atom37=residx_atom14_to_atom37,
        residx_atom37_to_atom14=residx_atom37_to_atom14,
        atom37_atom_exists=residx_atom37_mask,
    )
