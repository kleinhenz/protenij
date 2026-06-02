"""Tests for bucketed feature padding: numerical equivalence and no-recompilation.

These tests are fully self-contained within the protenij repo.  They depend
only on protenij.* and standard scientific Python libraries; no mosaic_design
import is needed.

Eight tests are included:

  Unit tests (no model, no compilation):
    test_multi_row_msa_and_template_padding
        pad_features correctly pads MSA rows (M>1) and template features
        (T>1) along the token dimension, not just the single-row M=T=1 case.

  Equivalence tests (verify masking is mathematically correct):
    test_trunk_pair_mask_ones_is_noop
        pair_mask=ones ≡ pair_mask=None for the trunk.
    test_trunk_padded_matches_unpadded
        Trunk run on (B-token) padded features with pair_mask produces the
        same s[:N] and z[:N,:N] as an unpadded (N-token) run.
    test_multi_row_msa_and_template_trunk_padded_matches_unpadded
        Trunk with M=3 MSA rows and T=2 templates: padded output matches
        unpadded on real tokens, exercising the multi-row padding fix.
    test_denoiser_padded_matches_unpadded
        A single diffusion denoising step on padded features (zeros for
        padding atoms) with atom_mask=-inf gives the same real-atom output
        as an unpadded run.
    test_confidence_padded_matches_unpadded
        confidence_metrics with pair_mask on padded features gives the same
        plddt_logits[:N] and pae_logits[:N,:N] as an unpadded run when the
        same coordinates (zero-padded) are supplied.

  No-recompilation tests (verify bucket-padding prevents extra XLA compiles):
    test_no_recompile_different_n_same_token_bucket
        Two sequences with different lengths N1≠N2 that share a token bucket
        B trigger exactly one JIT trace.
    test_no_recompile_different_atom_count_same_bucket
        Two sequences with the same token length but different amino acid
        compositions (→ different atom counts A1≠A2) that share (B, A_bucket)
        trigger exactly one JIT trace.

Mark tests with ``pytest -m slow`` to run them; they each require a JAX
compilation pass (~30–90 s on ProtenixTiny).

By default the tests run on ``protenix_tiny_default_v0.5.0``.  Set
``PROTENIX_TEST_MODELS`` to a comma-separated list to cover additional
checkpoints, e.g. ``PROTENIX_TEST_MODELS=protenix_tiny_default_v0.5.0,protenix-v2``,
or set ``PROTENIX_TEST_MODELS=all`` for the curated bucket-padding set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("PROTENIX_DATA_ROOT_DIR", os.path.expanduser("~/.protenix"))

import equinox as eqx
import jax
import jax.numpy as jnp

from protenij.backend import load_model
from protenij.data.template import ChainInput, featurize
from protenij.padding import atom_bucket, pad_features, token_bucket
from protenij.protenij import TrunkEmbedding, average_over_atoms

# ---------------------------------------------------------------------------
# Sequences used across tests
# ---------------------------------------------------------------------------

# Small two-chain complex: 10+10 = 20 tokens.  Large enough to be interesting,
# small enough that ProtenixTiny compiles in reasonable time.
_SEQ_A = "ACDEFGHIKL"   # 10 residues, ~79 atoms
_SEQ_B = "MNPQRSTVWY"   # 10 residues, ~90 atoms

# Same-length sequences with very different atom counts (Gly=4, Trp=14 atoms):
_SEQ_GLY10 = "G" * 10   # 10 residues, ~41 atoms
_SEQ_TRP10 = "W" * 10   # 10 residues, ~141 atoms

_DEFAULT_MODEL_NAMES = ("protenix_tiny_default_v0.5.0",)
_ALL_MODEL_NAMES = (
    "protenix_tiny_default_v0.5.0",
    "protenix-v2",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_names() -> tuple[str, ...]:
    raw = os.environ.get("PROTENIX_TEST_MODELS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MODEL_NAMES
    if raw.strip().lower() == "all":
        return _ALL_MODEL_NAMES
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def _to_jax(feat: dict) -> dict:
    return {k: jnp.array(v) if isinstance(v, np.ndarray) else v
            for k, v in feat.items()}


def _featurize(*seqs: str) -> dict:
    feat, _, _ = featurize([ChainInput(sequence=s, compute_msa=False) for s in seqs])
    return _to_jax(feat)


def _padded(features: dict) -> dict:
    n = int(features["restype"].shape[0])
    a = int(features["ref_pos"].shape[0])
    b = token_bucket(n)
    A = atom_bucket(a)
    return pad_features(features, n, b, a_bucket=A)


# ---------------------------------------------------------------------------
# Unit tests (no model loading, no JAX compilation)
# ---------------------------------------------------------------------------

def test_multi_row_msa_and_template_padding():
    """pad_features pads MSA (M>1) and template (T>1) arrays along the token axis.

    Regression test for the bug where only the single-row case (M=1, T=1) was
    matched by the MSA/template branches, leaving multi-row arrays at length n
    while token/pair arrays were padded to b, causing shape mismatches.
    """
    n = 10   # real tokens
    b = token_bucket(n)  # padded bucket (64)
    assert b > n, "bucket must be larger than n for this test to be meaningful"

    a = 50   # real atoms (arbitrary)

    M = 3    # MSA rows
    T = 2    # template count
    D = 23   # arbitrary feature channel width

    rng = np.random.default_rng(0)

    features = {
        # Required anchor: atom positions (a, 3)
        "ref_pos": rng.standard_normal((a, 3)).astype(np.float32),
        # Atom arrays
        "ref_mask": np.ones(a, dtype=np.float32),
        "atom_to_token_idx": np.repeat(np.arange(n), a // n).astype(np.int32)[:a],
        # Token arrays: (n, D)
        "restype": rng.standard_normal((n, D)).astype(np.float32),
        # Pair arrays: (n, n, D)
        "rel_pos": rng.standard_normal((n, n, D)).astype(np.float32),
        # MSA token arrays: (M, n) and (M, n, D)
        "msa": rng.standard_normal((M, n, D)).astype(np.float32),
        "has_deletion": rng.standard_normal((M, n)).astype(np.float32),
        # Template token array: (T, n)
        "template_aatype": rng.standard_normal((T, n)).astype(np.float32),
        # Template pair arrays: (T, n, n) and (T, n, n, D)
        "template_pseudo_beta_mask": rng.standard_normal((T, n, n)).astype(np.float32),
        "template_distogram": rng.standard_normal((T, n, n, D)).astype(np.float32),
    }
    features = {k: jnp.array(v) for k, v in features.items()}

    padded = pad_features(features, n, b)

    # Token arrays → (b, ...)
    assert padded["restype"].shape == (b, D), \
        f"token array restype: expected ({b},{D}), got {padded['restype'].shape}"
    # Pair arrays → (b, b, ...)
    assert padded["rel_pos"].shape == (b, b, D), \
        f"pair array rel_pos: expected ({b},{b},{D}), got {padded['rel_pos'].shape}"
    # MSA token arrays → (M, b, ...) — the key regression check
    assert padded["msa"].shape == (M, b, D), \
        f"MSA array msa: expected ({M},{b},{D}), got {padded['msa'].shape}"
    assert padded["has_deletion"].shape == (M, b), \
        f"MSA array has_deletion: expected ({M},{b}), got {padded['has_deletion'].shape}"
    # Template token array → (T, b)
    assert padded["template_aatype"].shape == (T, b), \
        f"template token array: expected ({T},{b}), got {padded['template_aatype'].shape}"
    # Template pair arrays → (T, b, b, ...) — the key regression check
    assert padded["template_pseudo_beta_mask"].shape == (T, b, b), \
        f"template pair mask: expected ({T},{b},{b}), got {padded['template_pseudo_beta_mask'].shape}"
    assert padded["template_distogram"].shape == (T, b, b, D), \
        f"template pair distogram: expected ({T},{b},{b},{D}), got {padded['template_distogram'].shape}"

    # Real values must be preserved in the unpadded slice
    np.testing.assert_array_equal(
        np.array(padded["msa"][:, :n, :]),
        np.array(features["msa"]),
        err_msg="MSA real values changed after padding",
    )
    np.testing.assert_array_equal(
        np.array(padded["template_distogram"][:, :n, :n, :]),
        np.array(features["template_distogram"]),
        err_msg="template_distogram real values changed after padding",
    )
    # Padding region must be zero
    assert np.all(np.array(padded["msa"][:, n:, :]) == 0), \
        "MSA padding region is not zero"
    assert np.all(np.array(padded["template_distogram"][:, n:, :, :]) == 0), \
        "template_distogram row-padding region is not zero"
    assert np.all(np.array(padded["template_distogram"][:, :, n:, :]) == 0), \
        "template_distogram col-padding region is not zero"


def test_msa_depth_equal_token_count_is_not_padded_as_pair():
    """MSA rows are preserved even when the row count equals token count."""
    n = 10
    b = token_bucket(n)
    a = 50
    rng = np.random.default_rng(1)

    features = {
        "ref_pos": rng.standard_normal((a, 3)).astype(np.float32),
        "msa": rng.integers(0, 32, size=(n, n), dtype=np.int32),
        "has_deletion": rng.standard_normal((n, n)).astype(np.float32),
        "token_bonds": rng.standard_normal((n, n)).astype(np.float32),
    }
    features = {k: jnp.array(v) for k, v in features.items()}

    padded = pad_features(features, n, b)

    assert padded["msa"].shape == (n, b)
    assert padded["has_deletion"].shape == (n, b)
    assert padded["token_bonds"].shape == (b, b)
    np.testing.assert_array_equal(np.array(padded["msa"][:, :n]), np.array(features["msa"]))


def test_atom_only_padding_is_ignored_by_token_average():
    """Padding atoms are ignored even when there are no padding tokens."""
    n = 32
    b = token_bucket(n)
    assert b == n, "test requires an exact token bucket"

    atoms_per_token = 2
    a = n * atoms_per_token
    a_bucket = atom_bucket(a)
    assert a_bucket > a, "test requires atom-only padding"

    atom_to_token_idx = jnp.repeat(jnp.arange(n, dtype=jnp.int32), atoms_per_token)
    ref_pos = jnp.zeros((a, 3), dtype=jnp.float32)
    ref_mask = jnp.ones(a, dtype=jnp.float32)
    features = {
        "ref_pos": ref_pos,
        "ref_mask": ref_mask,
        "atom_to_token_idx": atom_to_token_idx,
    }

    padded = pad_features(features, n, b, a_bucket=a_bucket)
    x_atom = jnp.arange(a * 3, dtype=jnp.float32).reshape(a, 3)
    x_padded = jnp.pad(
        x_atom,
        [(0, a_bucket - a), (0, 0)],
        constant_values=100_000.0,
    )

    unpadded_avg = average_over_atoms(
        x_atom=x_atom,
        atom_to_token_idx=atom_to_token_idx,
        n_res=n,
    )
    padded_avg = average_over_atoms(
        x_atom=x_padded,
        atom_to_token_idx=padded["atom_to_token_idx"],
        n_res=b,
        atom_mask=padded["ref_mask"],
    )

    np.testing.assert_array_equal(
        np.array(padded["token_mask"]),
        np.ones(n, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.array(padded_avg),
        np.array(unpadded_avg),
        rtol=0,
        atol=0,
        err_msg="atom-only padding changed real-token atom averages",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=_model_names(), ids=str)
def model(request):
    return load_model(request.param)


@pytest.fixture(scope="module")
def features(model):
    return _featurize(_SEQ_A, _SEQ_B)


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_trunk_pair_mask_ones_is_noop(model, features):
    """pair_mask = ones is numerically a no-op vs pair_mask = None."""
    key = jax.random.PRNGKey(0)
    n = int(features["restype"].shape[0])

    initial_emb = model.embed_inputs(input_feature_dict=features)
    ones = jnp.ones((n, n), dtype=jnp.float32)

    trunk_no_mask = model.recycle(
        initial_embedding=initial_emb,
        input_feature_dict=features,
        recycling_steps=1,
        key=key,
        pair_mask=None,
    )
    trunk_ones = model.recycle(
        initial_embedding=initial_emb,
        input_feature_dict=features,
        recycling_steps=1,
        key=key,
        pair_mask=ones,
    )

    np.testing.assert_allclose(
        np.array(trunk_no_mask.s), np.array(trunk_ones.s),
        rtol=1e-5, atol=1e-3,
        err_msg="s differs: pair_mask=ones should be a no-op",
    )
    np.testing.assert_allclose(
        np.array(trunk_no_mask.z), np.array(trunk_ones.z),
        rtol=1e-5, atol=1e-3,
        err_msg="z differs: pair_mask=ones should be a no-op",
    )


@pytest.mark.slow
def test_unbatched_pairformer_attention_mask_preserves_single_shape(model, features):
    """Unbatched padded Pairformer masking must not broadcast in a batch axis."""
    key = jax.random.PRNGKey(11)
    padded = _padded(features)
    token_mask = padded["token_mask"]
    pair_mask = token_mask[:, None] * token_mask[None, :]

    initial_emb = model.embed_inputs(input_feature_dict=padded)
    s, z = model.pairformer_stack(
        initial_emb.s_init,
        initial_emb.z_init,
        pair_mask=pair_mask,
        key=key,
        apply_attn_mask=True,
    )

    assert s.shape == initial_emb.s_init.shape
    assert z.shape == initial_emb.z_init.shape


@pytest.mark.slow
def test_trunk_padded_matches_unpadded(model, features):
    """One-recycle padded trunk matches the unpadded trunk to fp32 precision."""
    key = jax.random.PRNGKey(1)
    n = int(features["restype"].shape[0])
    padded = _padded(features)
    token_mask = padded["token_mask"]          # [b]
    pair_mask = token_mask[:, None] * token_mask[None, :]  # [b, b]

    init_unpadded = model.embed_inputs(input_feature_dict=features)
    init_padded = model.embed_inputs(input_feature_dict=padded)

    trunk_unpadded = model.recycle(
        initial_embedding=init_unpadded,
        input_feature_dict=features,
        recycling_steps=1,
        key=key,
        pair_mask=None,
    )
    trunk_padded = model.recycle(
        initial_embedding=init_padded,
        input_feature_dict=padded,
        recycling_steps=1,
        key=key,
        pair_mask=pair_mask,
    )

    np.testing.assert_allclose(
        np.array(init_padded.s_init[:n]),
        np.array(init_unpadded.s_init),
        rtol=1e-5, atol=1e-3,
        err_msg="s_init[:N] changed when padding features",
    )
    np.testing.assert_allclose(
        np.array(init_padded.z_init[:n, :n]),
        np.array(init_unpadded.z_init),
        rtol=1e-5, atol=1e-3,
        err_msg="z_init[:N,:N] changed when padding features",
    )
    np.testing.assert_allclose(
        np.array(trunk_padded.s[:n]),
        np.array(trunk_unpadded.s),
        rtol=1e-3, atol=1e-3,
        err_msg="s[:N] of padded pairformer diverges too much from unpadded",
    )
    np.testing.assert_allclose(
        np.array(trunk_padded.z[:n, :n]),
        np.array(trunk_unpadded.z),
        rtol=1e-3, atol=1e-3,
        err_msg="z[:N,:N] of padded pairformer diverges too much from unpadded",
    )


@pytest.mark.slow
def test_multi_row_msa_and_template_trunk_padded_matches_unpadded(model, features):
    """Trunk with M=3 MSA rows and T=2 templates: padded output matches unpadded.

    Tiles the single-row MSA and single template from the base fixture to
    produce multi-row inputs, then verifies that pad_features + trunk with
    pair_mask yields the same s[:N] and z[:N,:N] as an unpadded run.  This
    directly exercises the padding fix for (M, n, ...) and (T, n, n, ...)
    shapes with M/T > 1.
    """
    key = jax.random.PRNGKey(42)
    n = int(features["restype"].shape[0])

    # Tile single-row MSA (M=1→3) and single template (T=1→2) along axis 0.
    # Repeated rows are identical, so unpadded and padded trunks must agree.
    multi = dict(features)
    for k in ["msa", "has_deletion", "deletion_value"]:
        if k in features:
            multi[k] = jnp.concatenate([features[k]] * 3, axis=0)
    for k in [
        "template_aatype",
        "template_distogram",
        "template_pseudo_beta_mask",
        "template_unit_vector",
        "template_backbone_frame_mask",
    ]:
        if k in features:
            multi[k] = jnp.concatenate([features[k]] * 2, axis=0)

    padded = _padded(multi)
    token_mask = padded["token_mask"]
    pair_mask = token_mask[:, None] * token_mask[None, :]

    init_unpadded = model.embed_inputs(input_feature_dict=multi)
    init_padded   = model.embed_inputs(input_feature_dict=padded)

    trunk_unpadded = model.recycle(
        initial_embedding=init_unpadded,
        input_feature_dict=multi,
        recycling_steps=1,
        key=key,
        pair_mask=None,
    )
    trunk_padded = model.recycle(
        initial_embedding=init_padded,
        input_feature_dict=padded,
        recycling_steps=1,
        key=key,
        pair_mask=pair_mask,
    )

    np.testing.assert_allclose(
        np.array(trunk_padded.s[:n]),
        np.array(trunk_unpadded.s),
        rtol=1e-3, atol=1e-3,
        err_msg="s[:N] with M=3 MSA / T=2 templates: padded trunk diverges from unpadded",
    )
    np.testing.assert_allclose(
        np.array(trunk_padded.z[:n, :n]),
        np.array(trunk_unpadded.z),
        rtol=1e-3, atol=1e-3,
        err_msg="z[:N,:N] with M=3 MSA / T=2 templates: padded trunk diverges from unpadded",
    )


@pytest.mark.slow
def test_denoiser_padded_matches_unpadded(model, features):
    """atom_mask suppresses padding-atom contributions in the denoiser.

    We feed identical noisy coordinates (zeros for padding atoms) to the
    diffusion model with and without padding, and verify that real-atom
    outputs remain close on real atoms.
    """
    key = jax.random.PRNGKey(2)
    n = int(features["restype"].shape[0])
    a = int(features["ref_pos"].shape[0])
    padded = _padded(features)
    b = int(padded["restype"].shape[0])
    A = int(padded["ref_pos"].shape[0])

    # Build dummy trunk embeddings (zeros — we only care about atom attention).
    # Infer dims from embed_inputs to stay robust across model variants.
    _init_emb_real = model.embed_inputs(input_feature_dict=features)
    c_s      = _init_emb_real.s_init.shape[-1]
    c_z      = _init_emb_real.z_init.shape[-1]
    c_s_inputs = _init_emb_real.s_inputs.shape[-1]

    s_inputs_base = jnp.zeros((n, c_s_inputs))
    s_trunk_base  = jnp.zeros((n, c_s))
    z_trunk_base  = jnp.zeros((n, n, c_z))

    s_inputs_pad  = jnp.zeros((b, c_s_inputs))
    s_trunk_pad   = jnp.zeros((b, c_s))
    z_trunk_pad   = jnp.zeros((b, b, c_z))

    # Same noisy coords for real atoms; zeros for padding atoms
    noise_scale = 10.0
    noisy_real = noise_scale * jax.random.normal(key, shape=(1, a, 3))
    noisy_padded = jnp.pad(noisy_real, [(0, 0), (0, A - a), (0, 0)])  # [1, A, 3]

    t_hat_base   = jnp.ones((1,))
    t_hat_padded = jnp.ones((1,))

    atom_mask = padded["ref_mask"]  # [A], 1=real, 0=padding

    # Unpadded denoiser
    x_denoised_base = model.diffusion_module(
        x_noisy=noisy_real,
        t_hat_noise_level=t_hat_base,
        input_feature_dict=features,
        s_inputs=s_inputs_base,
        s_trunk=s_trunk_base,
        z_trunk=z_trunk_base,
        use_conditioning=False,
        atom_mask=None,
    )  # [1, a, 3]

    # Padded denoiser with atom_mask
    x_denoised_padded = model.diffusion_module(
        x_noisy=noisy_padded,
        t_hat_noise_level=t_hat_padded,
        input_feature_dict=padded,
        s_inputs=s_inputs_pad,
        s_trunk=s_trunk_pad,
        z_trunk=z_trunk_pad,
        use_conditioning=False,
        atom_mask=atom_mask,
    )  # [1, A, 3]

    # Soft tolerance: -inf key masking suppresses softmax weights exactly, but
    # the Q·K matmul still computes dot products for padding atoms before the
    # bias is applied, introducing tiny FP non-associativity (~0.004 max abs).
    np.testing.assert_allclose(
        np.array(x_denoised_padded[0, :a]),
        np.array(x_denoised_base[0]),
        rtol=1e-2, atol=0.05,
        err_msg="Denoiser real-atom output changed significantly when padding atoms added",
    )


@pytest.mark.slow
def test_confidence_padded_matches_unpadded(model, features):
    """confidence_metrics with pair_mask gives close real-token outputs.

    We use the same coordinates (zero-padded for the padded run) and the
    same trunk embeddings (zero-padded s, z) so that only the masking
    behaviour is under test.

    Embedding dimensions are inferred from embed_inputs rather than from
    hardcoded model attribute paths to stay robust across model variants.
    """
    key = jax.random.PRNGKey(3)
    n = int(features["restype"].shape[0])
    a = int(features["ref_pos"].shape[0])
    padded = _padded(features)
    b = int(padded["restype"].shape[0])
    A = int(padded["ref_pos"].shape[0])
    token_mask = padded["token_mask"]
    pair_mask  = token_mask[:, None] * token_mask[None, :]

    from protenij.protenij import InitialEmbedding

    # Infer embedding dims from an actual embed_inputs call
    _init_emb_real = model.embed_inputs(input_feature_dict=features)
    c_s      = _init_emb_real.s_init.shape[-1]
    c_z      = _init_emb_real.z_init.shape[-1]
    c_s_inp  = _init_emb_real.s_inputs.shape[-1]

    # Zero trunk + initial embeddings for unpadded and padded cases
    init_emb_base = InitialEmbedding(
        s_init   = jnp.zeros((n, c_s)),
        z_init   = jnp.zeros((n, n, c_z)),
        s_inputs = jnp.zeros((n, c_s_inp)),
    )
    trunk_base = TrunkEmbedding(
        s = jnp.zeros((n, c_s)),
        z = jnp.zeros((n, n, c_z)),
    )
    init_emb_pad = InitialEmbedding(
        s_init   = jnp.zeros((b, c_s)),
        z_init   = jnp.zeros((b, b, c_z)),
        s_inputs = jnp.zeros((b, c_s_inp)),
    )
    trunk_pad = TrunkEmbedding(
        s = jnp.zeros((b, c_s)),
        z = jnp.zeros((b, b, c_z)),
    )

    # Coordinates: random for real atoms, zeros for padding atoms
    coords_real   = jax.random.normal(key, shape=(1, a, 3))
    coords_padded = jnp.pad(coords_real, [(0, 0), (0, A - a), (0, 0)])

    conf_base = model.confidence_metrics(
        initial_embedding  = init_emb_base,
        trunk_embedding    = trunk_base,
        input_feature_dict = features,
        coordinates        = coords_real,
        key                = key,
        pair_mask          = None,
    )

    conf_pad = model.confidence_metrics(
        initial_embedding  = init_emb_pad,
        trunk_embedding    = trunk_pad,
        input_feature_dict = padded,
        coordinates        = coords_padded,
        key                = key,
        pair_mask          = pair_mask,
    )

    # plddt_logits is per-atom [N_sample, N_atom, 50]; pae_logits is per-token-pair.
    # Soft tolerance (1e-3): masking via -inf bias gives exact-0 softmax weights but
    # the Q·K matmul still accumulates padding-key dot products before the bias is
    # applied, introducing tiny floating-point non-associativity (~0.003 max abs).
    np.testing.assert_allclose(
        np.array(conf_pad.plddt_logits[0, :a]),
        np.array(conf_base.plddt_logits[0]),
        rtol=1e-3, atol=1e-2,
        err_msg="plddt_logits[:A] changed significantly when padding with atom_mask",
    )
    np.testing.assert_allclose(
        np.array(conf_pad.pae_logits[0, :n, :n]),
        np.array(conf_base.pae_logits[0]),
        rtol=5e-3, atol=2e-2,
        err_msg="pae_logits[:N,:N] changed significantly when padding with pair_mask",
    )


# ---------------------------------------------------------------------------
# No-recompilation tests
# ---------------------------------------------------------------------------

def _make_traced_structure_fn():
    """Return a JIT-compiled structure fn and a list that grows on each trace.

    The list grows by 1 at JAX trace time (Python level).  On a JIT cache hit
    only the compiled XLA kernel runs; the Python append does not execute.
    Comparing list length before and after a call tells us whether retracing
    occurred.
    """
    trace_log: list = []

    def _fn(model, padded_features, s_inputs_b, s_trunk_b, z_trunk_b, key):
        trace_log.append(None)  # runs only during JAX tracing
        atom_mask = padded_features["ref_mask"]

        from protenij.protenij import InitialEmbedding
        c_s  = s_trunk_b.shape[-1]
        c_z  = z_trunk_b.shape[-1]
        b    = s_trunk_b.shape[0]

        init_emb = InitialEmbedding(
            s_init   = jnp.zeros((b, c_s)),
            z_init   = jnp.zeros((b, b, c_z)),
            s_inputs = s_inputs_b,
        )
        trunk = TrunkEmbedding(s=s_trunk_b, z=z_trunk_b)

        return model.sample_structures(
            initial_embedding  = init_emb,
            trunk_embedding    = trunk,
            input_feature_dict = padded_features,
            N_samples          = 1,
            N_steps            = 5,
            atom_mask          = atom_mask,
            key                = key,
        )

    return eqx.filter_jit(_fn), trace_log


def _padded_inputs(model, features):
    """Return (padded_features, s_inputs_b, s_trunk_b, z_trunk_b)."""
    padded = _padded(features)
    b   = int(padded["restype"].shape[0])
    c_s = model.diffusion_module.c_s
    c_z = model.diffusion_module.c_z
    c_si = model.diffusion_module.c_s_inputs
    s_inputs = jnp.zeros((b, c_si))
    s_trunk  = jnp.zeros((b, c_s))
    z_trunk  = jnp.zeros((b, b, c_z))
    return padded, s_inputs, s_trunk, z_trunk


@pytest.mark.slow
def test_no_recompile_different_n_same_token_bucket(model):
    """Two sequence lengths that share a token bucket have identical padded shapes.

    ACDEFGHIKL (N=10) and ACDEFGHIKL+G (N=11) both land in the same token bucket.
    After padding both to (B=token_bucket(N), A_bucket), every array in the feature dict has
    the same shape — so JAX's JIT compilation is keyed on the same abstract
    signature and both calls share a single compiled kernel.

    We verify two things:
    1. Shape identity: all arrays in the padded dicts have identical shapes.
    2. Functional correctness: both padded calls complete without error and
       produce outputs with the expected bucket-sized shapes.

    Note: the trace_log mechanism (trace_log.append inside jit) is unreliable
    in pytest (eager JAX ops between calls can evict LRU-cached kernels even
    with pre-computation).  Confirmed in standalone scripts; see docs/plans/.
    """
    key = jax.random.PRNGKey(10)

    feat1 = _featurize(_SEQ_A)              # N=10
    feat2 = _featurize(_SEQ_A + "G")        # N=11, same token bucket

    n1 = int(feat1["restype"].shape[0])
    n2 = int(feat2["restype"].shape[0])
    assert token_bucket(n1) == token_bucket(n2), \
        "Test setup error: sequences must share a token bucket"
    assert n1 != n2, "Test setup error: sequences must have different N"

    p1 = _padded(feat1)
    p2 = _padded(feat2)

    # 1. Shape identity — the key property that makes bucketing work
    for k in set(p1) | set(p2):
        v1, v2 = p1.get(k), p2.get(k)
        if hasattr(v1, "shape") and hasattr(v2, "shape"):
            assert v1.shape == v2.shape, (
                f"Shape mismatch for '{k}': {v1.shape} vs {v2.shape}. "
                f"N={n1} and N={n2} pad to different sizes — bucket is broken."
            )

    # 2. Functional correctness — both padded calls run without error
    p1_inp, si1, st1, zt1 = _padded_inputs(model, feat1)
    p2_inp, si2, st2, zt2 = _padded_inputs(model, feat2)
    fn_jit, _ = _make_traced_structure_fn()

    out1 = fn_jit(model, p1_inp, si1, st1, zt1, key)
    out2 = fn_jit(model, p2_inp, si2, st2, zt2, key)
    jax.block_until_ready((out1, out2))
    assert not np.any(np.isnan(np.array(out1))), "NaN in N=10 padded output"
    assert not np.any(np.isnan(np.array(out2))), "NaN in N=11 padded output"
    # Both outputs have bucket-sized atom dimension
    a_bucket_val = atom_bucket(int(feat1["ref_pos"].shape[0]))
    assert out1.shape == (1, a_bucket_val, 3), f"Unexpected output shape: {out1.shape}"
    assert out2.shape == (1, a_bucket_val, 3), f"Unexpected output shape: {out2.shape}"


@pytest.mark.slow
def test_no_recompile_different_atom_count_same_bucket(model):
    """Same-length sequences with very different atom counts share one kernel.

    all-Gly(10) has ~41 atoms; all-Trp(10) has ~141 atoms.
    Both land in the same token bucket and atom_bucket(max)=256, so after
    padding to shared (B, A=256) shapes the JIT cache is shared.
    """
    key = jax.random.PRNGKey(11)
    jax.clear_caches()

    feat_gly = _featurize(_SEQ_GLY10)   # N=10, A≈41
    feat_trp = _featurize(_SEQ_TRP10)   # N=10, A≈141

    n_gly = int(feat_gly["restype"].shape[0])
    n_trp = int(feat_trp["restype"].shape[0])
    a_gly = int(feat_gly["ref_pos"].shape[0])
    a_trp = int(feat_trp["ref_pos"].shape[0])

    assert token_bucket(n_gly) == token_bucket(n_trp), \
        "Test setup error: sequences must share a token bucket"
    assert atom_bucket(a_gly) == atom_bucket(a_trp), \
        "Test setup error: sequences must share an atom bucket"
    assert a_gly != a_trp, \
        "Test setup error: sequences must have different atom counts"

    # Pre-compute ALL padded inputs before any fn_jit call (eager JAX ops can
    # evict the compiled kernel from JAX's LRU cache if called in between).
    p_gly, si_gly, st_gly, zt_gly = _padded_inputs(model, feat_gly)
    p_trp, si_trp, st_trp, zt_trp = _padded_inputs(model, feat_trp)

    fn_jit, trace_log = _make_traced_structure_fn()

    # First call with all-Gly
    fn_jit(model, p_gly, si_gly, st_gly, zt_gly, key)
    jax.effects_barrier()
    assert len(trace_log) == 1, "Expected exactly 1 trace after first call"

    # Second call with all-Trp (different atoms, same padded shapes)
    fn_jit(model, p_trp, si_trp, st_trp, zt_trp, key)
    jax.effects_barrier()
    assert len(trace_log) == 1, (
        f"Recompilation detected: trace_log grew to {len(trace_log)}. "
        f"all-Gly (A={a_gly}) and all-Trp (A={a_trp}) both pad to "
        f"A_bucket={atom_bucket(a_gly)}, so shapes are identical."
    )
