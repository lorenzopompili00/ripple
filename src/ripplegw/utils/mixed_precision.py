"""Tools to run parts of a JAX waveform model in float32 inside a float64 process.

The intended split for frequency-domain models is:

* everything computed once per binary (fits, small linear solves, precession
  constants) stays in float64, where it is cheap and where float32 is often
  not accurate enough;
* everything evaluated per frequency (the large arrays) runs in float32.

Three tools are provided:

* :func:`cast` casts the floating leaves of a pytree to 32 or 64 bit.
* :func:`run_in` calls a function with its inputs cast to 32 or 64 bit; for 32
  bit the trace runs with ``jax.enable_x64(False)``, so Python float
  constants, ``jnp.array`` literals and ``jnp.pi`` inside the function become
  float32 exactly as in a process without x64.
* :func:`split_by_frequency` takes a function ``fn(x, tree)`` of a frequency
  array ``x`` and per-binary data ``tree`` and splits its jaxpr by data
  dependence on ``x`` (``partial_eval_jaxpr_nounits``): the part that does not
  depend on ``x`` runs in float64, the part that does runs in float32. This
  handles code where the per-binary and per-frequency work are interleaved
  (e.g. the XHM mode coefficients) without rewriting it.

All of this requires ``jax_enable_x64=True`` in the process
(:func:`require_x64`).

Private JAX APIs. :func:`split_by_frequency` and :func:`eval_jaxpr_32` use
JAX internals, tested with jax 0.12:

* ``jax._src.core`` (``Literal``, ``eval_jaxpr``, ``Jaxpr``; ``ClosedJaxpr``
  where it still exists);
* ``jax._src.interpreters.partial_eval`` (``partial_eval_jaxpr_nounits``,
  ``convert_constvars_jaxpr``);
* the equation parameter layout of ``jit``/``pjit`` (``jaxpr``),
  ``closed_call``/``core_call``/``custom_jvp_call``/``custom_vjp_call``
  (``call_jaxpr``), ``cond`` (``branches``) and ``while`` (``cond_jaxpr``,
  ``body_jaxpr``, ``cond_nconsts``, ``body_nconsts``).

Only those higher-order primitives are supported; any other primitive that
carries a sub-jaxpr (``scan``, ``remat``, ``shard_map``, ...) raises
``NotImplementedError`` instead of being re-bound with a 64-bit body.
``custom_jvp``/``custom_vjp`` rules are inlined: gradients through the float32
part differentiate the primal implementation. ``tests/unit/utils/
test_mixed_precision.py`` pins this behaviour.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["cast", "eval_jaxpr_32", "require_x64", "run_in", "split_by_frequency"]


def require_x64(what: str = "mixed precision") -> None:
    """Raise a clear error if 64-bit JAX is not enabled."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} needs float64 for the per-binary stages: enable it with "
            "jax.config.update('jax_enable_x64', True) before creating arrays."
        )


def cast(tree: Any, bits: int) -> Any:
    """Cast every floating / complex leaf (and Python float) of ``tree`` to ``bits`` (32 or 64)."""
    fdt, cdt = (
        (jnp.float32, jnp.complex64) if bits == 32 else (jnp.float64, jnp.complex128)
    )

    def c(x):
        if isinstance(x, float):
            return jnp.asarray(x, fdt)
        if hasattr(x, "dtype"):
            if jnp.issubdtype(x.dtype, jnp.complexfloating):
                return jnp.asarray(x).astype(cdt)
            if jnp.issubdtype(x.dtype, jnp.floating):
                return jnp.asarray(x).astype(fdt)
        return x

    return jax.tree_util.tree_map(c, tree)


def run_in(bits: int, fn: Callable, *args: Any) -> Any:
    """``fn(*args)`` with all floating inputs cast to ``bits``; for 32 the trace runs with x64 off."""
    if bits == 64:
        return fn(*cast(args, 64))
    args = cast(args, 32)
    with jax.enable_x64(False):
        return fn(*args)


_TO32 = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("complex128"): np.dtype("complex64"),
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
}


def _val32(x):
    if not isinstance(x, (np.ndarray, np.generic, jax.Array)) and hasattr(x, "dtype"):
        x = np.asarray(x)  # literal value classes (e.g. typed Python scalars)
    if hasattr(x, "dtype"):
        nd = _TO32.get(np.dtype(x.dtype), np.dtype(x.dtype))
        if nd != x.dtype:
            if isinstance(x, (np.ndarray, np.generic)):
                return np.asarray(x, nd)
            return x.astype(nd)
    return x


def _params32(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, np.dtype) or (
            isinstance(v, type) and issubclass(v, np.generic)
        ):
            v = _TO32.get(np.dtype(v), np.dtype(v))
        out[k] = v
    return out


def _has_subjaxpr(params: dict) -> bool:
    from jax._src import core

    jaxpr_types = tuple({core.Jaxpr, getattr(core, "ClosedJaxpr", core.Jaxpr)})

    def is_jaxpr(v):
        return isinstance(v, jaxpr_types)

    return any(
        is_jaxpr(v) or (isinstance(v, (list, tuple)) and any(map(is_jaxpr, v)))
        for v in params.values()
    )


def eval_jaxpr_32(jaxpr, consts, *args):
    """Evaluate a (64-bit) jaxpr in 32 bit.

    Every value is cast to float32 / complex64 / int32 and every primitive is
    re-bound with 32-bit parameters (call under ``jax.enable_x64(False)``).
    Calls (jit, custom_jvp/vjp) are inlined; cond / while are rebuilt.
    """
    from jax._src import core

    _supported = {
        "jit",
        "pjit",
        "closed_call",
        "core_call",
        "custom_jvp_call",
        "custom_vjp_call",
        "cond",
        "while",
    }
    env = {}

    def read(v):
        return _val32(v.val) if isinstance(v, core.Literal) else env[v]

    for v, c in zip(jaxpr.constvars, consts):
        env[v] = _val32(c)
    for v, a in zip(jaxpr.invars, args):
        env[v] = _val32(a)
    for e in jaxpr.eqns:
        inv = [read(v) for v in e.invars]
        name = e.primitive.name
        if name in ("jit", "pjit", "closed_call", "core_call"):
            cj = e.params.get("jaxpr", e.params.get("call_jaxpr"))
            outs = eval_jaxpr_32(cj.jaxpr, cj.consts, *inv)
        elif name in ("custom_jvp_call", "custom_vjp_call"):
            cj = e.params["call_jaxpr"]
            outs = eval_jaxpr_32(cj.jaxpr, cj.consts, *inv)
        elif name == "cond":
            branches = [
                (lambda *o, b=b: eval_jaxpr_32(b.jaxpr, b.consts, *o))
                for b in e.params["branches"]
            ]
            outs = lax.switch(inv[0], branches, *inv[1:])
        elif name == "while":
            cn, bn = e.params["cond_nconsts"], e.params["body_nconsts"]
            cc, bc, init = inv[:cn], inv[cn : cn + bn], inv[cn + bn :]
            cj, bj = e.params["cond_jaxpr"], e.params["body_jaxpr"]
            outs = list(
                lax.while_loop(
                    lambda c, cj=cj, cc=cc: eval_jaxpr_32(cj.jaxpr, cj.consts, *cc, *c)[
                        0
                    ],
                    lambda c, bj=bj, bc=bc: tuple(
                        eval_jaxpr_32(bj.jaxpr, bj.consts, *bc, *c)
                    ),
                    tuple(init),
                )
            )
        elif _has_subjaxpr(e.params):
            raise NotImplementedError(
                f"eval_jaxpr_32 / split_by_frequency: primitive {name!r} carries a "
                f"sub-jaxpr and is not supported (supported: {sorted(_supported)})"
            )
        else:
            outs = e.primitive.bind(*inv, **_params32(e.params))
        if not e.primitive.multiple_results:
            outs = [outs]
        for v, o in zip(e.outvars, outs):
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def split_by_frequency(fn: Callable, x_like: Any, tree_like: Any) -> Callable:
    """Split ``fn(x, tree)`` into a float64 part (independent of ``x``) and a float32 part.

    Args:
        fn: function of a frequency array ``x`` and a pytree of per-binary data.
        x_like, tree_like: example arguments (shapes / dtypes are used, values are not).

    Returns:
        ``g(x, tree)`` with the same outputs (32 bit). ``g.n_known`` / ``g.n_unknown``
        count the equations evaluated in 64 / 32 bit.
    """
    from jax._src import core
    from jax._src.interpreters import partial_eval as pe

    leaves, tdef = jax.tree_util.tree_flatten(tree_like)

    def sds(x):
        return jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x))

    def flat(x, *lv):
        return fn(x, jax.tree_util.tree_unflatten(tdef, lv))

    cj = jax.make_jaxpr(flat)(sds(x_like), *[sds(v) for v in leaves])
    out_tree = jax.tree_util.tree_structure(
        jax.eval_shape(flat, sds(x_like), *[sds(v) for v in leaves])
    )
    jx = pe.convert_constvars_jaxpr(cj.jaxpr)
    consts = list(cj.consts)
    unknowns = [False] * len(consts) + [True] + [False] * len(leaves)
    jk, ju, out_uk, res = pe.partial_eval_jaxpr_nounits(jx, unknowns, instantiate=True)
    if not all(out_uk):
        raise ValueError("every output of fn must depend on the frequency array")
    jk_j, jk_c = (jk.jaxpr, jk.consts) if hasattr(jk, "consts") else (jk, ())
    ju_j, ju_c = (ju.jaxpr, ju.consts) if hasattr(ju, "consts") else (ju, ())
    n_res = len(res)

    def g(x, tree):
        lv = jax.tree_util.tree_leaves(cast(tree, 64))
        r = core.eval_jaxpr(jk_j, jk_c, *consts, *lv)[-n_res:] if n_res else []
        with jax.enable_x64(False):
            outs = eval_jaxpr_32(ju_j, ju_c, *r, x)
        return jax.tree_util.tree_unflatten(out_tree, outs)

    g.n_known, g.n_unknown, g.n_res = len(jk_j.eqns), len(ju_j.eqns), n_res
    return g
