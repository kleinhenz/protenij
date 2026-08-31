"""Bucketed feature padding for JAX recompilation reduction.

JAX (via equinox's filter_jit) recompiles whenever input shapes change.
Padding features to fixed bucket sizes keeps shapes stable across calls
that differ only in sequence length or amino acid composition, so each
(token_bucket, atom_bucket) pair compiles exactly once.

This module is the canonical implementation.  Downstream consumers
(e.g. mosaic_design._backend) import from here.

Public API
----------
TOKEN_BUCKETS, ATOM_BUCKETS      — bucket boundary tables
token_bucket(n), atom_bucket(a)  — lookup helpers
pad_features(features, n, b, a_bucket=None)  — pad a feature dict
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Bucket tables
# ---------------------------------------------------------------------------

# Multiples of 32 up to 1024, then coarser above that.
TOKEN_BUCKETS: tuple[int, ...] = (
    32, 64, 96, 128, 160, 192, 224, 256,
    320, 384, 448, 512,
    640, 768, 896, 1024,
    1280, 1536, 2048,
)

# Multiples of 128 up to 2048, then coarser above that.
# Typical proteins run 5-10 atoms/residue; 128-step spacing gives good
# reuse without excessive padding for systems up to ~2000 atoms.
ATOM_BUCKETS: tuple[int, ...] = (
    256, 384, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048,
    2560, 3072, 3584, 4096,
    5120, 6144, 8192,
)


def token_bucket(n: int) -> int:
    """Smallest TOKEN_BUCKETS entry >= n."""
    for b in TOKEN_BUCKETS:
        if b >= n:
            return b
    return n  # larger than all buckets — caller decides what to do


def atom_bucket(a: int) -> int:
    """Smallest ATOM_BUCKETS entry >= a."""
    for b in ATOM_BUCKETS:
        if b >= a:
            return b
    return a


# ---------------------------------------------------------------------------
# Feature padding
# ---------------------------------------------------------------------------

def pad_features(features: dict, n: int, b: int, a_bucket: int | None = None) -> dict:
    """Pad token-level and atom-level feature arrays to fixed bucket sizes.

    Feature kinds are selected explicitly by feature name.  This avoids
    confusing leading MSA/template rows with token-pair axes when their sizes
    happen to match the token count.

    Token-level arrays get K=b-n zero rows appended; pair arrays get K rows
    and columns; atom arrays are padded to ``a_bucket`` atoms. When
    ``a_bucket`` is omitted, the smallest atom bucket that has room for all
    real atoms plus at least one atom for every padded token is selected.

    ``token_mask`` and ``atom_padding_mask`` keys (1s for real entries, 0s for
    padding) are added to the returned dict. Unlike ``ref_mask``, the latter is
    strictly padding metadata: a real atom remains 1 even when its reference
    conformer is unavailable.

    Args:
        features: Feature dict as produced by protenix featurization.
        n: Real token count.
        b: Target token bucket size (>= n).
        a_bucket: Optional target atom bucket override. Atom arrays are padded
            to exactly this size, which must satisfy
            ``a_bucket >= a + (b - n)``. When omitted, the target is computed
            as ``atom_bucket(a + (b - n))``.

    Returns:
        New dict with padded arrays, ``token_mask``, and
        ``atom_padding_mask`` keys.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if b < n:
        raise ValueError(f"b must be greater than or equal to n, got b={b}, n={n}")

    k = b - n
    a = features["ref_pos"].shape[0]  # static during JIT trace

    if a_bucket is None:
        a_bucket = atom_bucket(a + k)
    if n == b and a_bucket == a:
        return features

    assert a_bucket >= a, f"a_bucket ({a_bucket}) must be >= a ({a})"
    k_atoms = a_bucket - a
    assert k_atoms >= k, (
        f"atom bucket too small: {k_atoms} extra atom slots for {k} padding tokens"
    )

    def _pad(arr: jax.Array, axis: int, size: int) -> jax.Array:
        pad_width = [(0, size) if i == axis else (0, 0) for i in range(arr.ndim)]
        return jnp.pad(arr, pad_width)

    def _pad_axes(
        arr: jax.Array,
        axes: tuple[int, ...],
        size: int,
        expected_size: int,
        feature_name: str,
    ) -> jax.Array:
        for axis in axes:
            if axis >= arr.ndim:
                raise ValueError(
                    f"feature {feature_name!r} has shape {arr.shape}, "
                    f"but padding axis {axis} was requested"
                )
            if arr.shape[axis] != expected_size:
                raise ValueError(
                    f"feature {feature_name!r} has shape {arr.shape}; "
                    f"axis {axis} must have size {expected_size}"
                )
            arr = _pad(arr, axis, size)
        return arr

    def _pad_constraint_leaf(leaf):
        if not hasattr(leaf, "shape"):
            return leaf
        if leaf.ndim < 2 or leaf.shape[:2] != (n, n):
            raise ValueError(
                "constraint feature leaves must start with token-pair "
                f"axes ({n}, {n}); got {leaf.shape}"
            )
        return _pad_axes(
            leaf,
            (0, 1),
            k,
            n,
            "constraint_feature",
        )

    result: dict = {}
    for key_name, val in features.items():
        if key_name == "constraint_feature":
            result[key_name] = jax.tree.map(_pad_constraint_leaf, val)
            continue

        if not hasattr(val, "shape"):
            result[key_name] = val
            continue

        s = val.shape

        if s in ((), (1,)):
            result[key_name] = val

        elif key_name == "atom_to_token_idx":
            if s[0] != a:
                raise ValueError(
                    f"feature {key_name!r} has shape {s}; axis 0 must have size {a}"
                )
            # Map padding atoms to in-bounds token indices. ref_mask marks them
            # as padding, so mask-aware atom aggregation ignores them.
            if k > 0:
                # Cyclic round-robin: atom i goes to padding token (i % k).
                # This keeps padding atoms away from real tokens when padding
                # token slots exist.
                pad_ids = (jnp.arange(k_atoms, dtype=val.dtype) % k) + n
            else:
                # No padding tokens; use an in-bounds placeholder index.
                pad_ids = jnp.full(k_atoms, n - 1, dtype=val.dtype)
            result[key_name] = jnp.concatenate([val, pad_ids])

        elif key_name in TOKEN_AXIS0_FEATURES:
            result[key_name] = _pad_axes(val, (0,), k, n, key_name)

        elif key_name in TOKEN_PAIR_AXES01_FEATURES:
            result[key_name] = _pad_axes(val, (0, 1), k, n, key_name)

        elif key_name in TOKEN_AXIS1_FEATURES:
            result[key_name] = _pad_axes(val, (1,), k, n, key_name)

        elif key_name in TOKEN_PAIR_AXES12_FEATURES:
            result[key_name] = _pad_axes(val, (1, 2), k, n, key_name)

        elif key_name in ATOM_AXIS0_FEATURES:
            result[key_name] = _pad_axes(val, (0,), k_atoms, a, key_name)

        elif key_name in ATOM_PAIR_AXES01_FEATURES:
            result[key_name] = _pad_axes(val, (0, 1), k_atoms, a, key_name)

        else:
            result[key_name] = val

    result["token_mask"] = jnp.concatenate([
        jnp.ones(n, dtype=jnp.float32),
        jnp.zeros(k, dtype=jnp.float32),
    ])
    result["atom_padding_mask"] = jnp.concatenate([
        jnp.ones(a, dtype=jnp.float32),
        jnp.zeros(k_atoms, dtype=jnp.float32),
    ])
    return result


# Padding specs are intentionally key-based.  Several valid feature arrays have
# shape (M, N, ...) or (T, N, ...) where M or T can equal N, so using only the
# number and size of dimensions can misclassify them as pair features.
TOKEN_AXIS0_FEATURES: frozenset[str] = frozenset({
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "restype",
    "profile",
    "deletion_mean",
    "has_frame",
    "frame_atom_index",
    "atom_rep_atom_idx",
})

TOKEN_PAIR_AXES01_FEATURES: frozenset[str] = frozenset({
    "token_bonds",
    "centre_centre_distance",
    "centre_centre_distance_mask",
    "rel_pos",  # synthetic/test feature; real pair features use token_bonds.
})

TOKEN_AXIS1_FEATURES: frozenset[str] = frozenset({
    "msa",
    "has_deletion",
    "deletion_value",
    "template_aatype",
    "template_restype",
    "template_all_atom_mask",
    "template_all_atom_positions",
})

TOKEN_PAIR_AXES12_FEATURES: frozenset[str] = frozenset({
    "template_distogram",
    "template_pseudo_beta_mask",
    "template_unit_vector",
    "template_backbone_frame_mask",
})

ATOM_AXIS0_FEATURES: frozenset[str] = frozenset({
    "ref_pos",
    "ref_mask",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_to_tokatom_idx",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
    "is_ligand",
    "is_dna",
    "is_rna",
    "is_protein",
    "modified_res_mask",
    "distogram_rep_atom_mask",
    "pae_rep_atom_mask",
    "plddt_m_rep_atom_mask",
    "coordinate",
    "coordinate_mask",
    "centre_atom_mask",
})

ATOM_PAIR_AXES01_FEATURES: frozenset[str] = frozenset({
    "bond_mask",
})
