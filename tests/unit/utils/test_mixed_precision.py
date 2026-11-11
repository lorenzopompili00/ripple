"""ripplegw.utils.mixed_precision on small functions (pins the supported JAX features)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.utils import mixed_precision as mp

X = jnp.linspace(0.1, 1.0, 7)
TREE = {"a": jnp.asarray(0.3), "b": jnp.asarray(2.0)}


def test_cast_and_run_in():
    t32 = mp.cast({"x": X, "z": X * 1j, "n": jnp.arange(3), "f": 1.5}, 32)
    assert t32["x"].dtype == jnp.float32 and t32["z"].dtype == jnp.complex64
    assert t32["n"].dtype == jnp.arange(3).dtype and t32["f"].dtype == jnp.float32
    assert mp.run_in(32, lambda x: x * jnp.pi, X).dtype == jnp.float32
    assert (
        mp.run_in(64, lambda x: x * jnp.pi, X.astype(jnp.float32)).dtype == jnp.float64
    )


@pytest.mark.parametrize("use_jit", [False, True])
def test_split_supported_features(use_jit):
    """where, cond, nested jit, custom_jvp, reductions: f32 result close to f64."""

    @jax.custom_jvp
    def softplus(x):
        return jnp.log1p(jnp.exp(x))

    softplus.defjvp(lambda p, t: (softplus(p[0]), t[0] * jax.nn.sigmoid(p[0])))

    @jax.jit
    def inner(x, b):
        return jnp.sin(x) * b

    def f(x, t):
        coeff = jnp.cos(t["a"]) * t["b"] ** 2  # per-binary: float64 side
        y = jnp.where(x > t["a"], inner(x, t["b"]), x**2) * coeff
        y = jax.lax.cond(t["a"] > 0, lambda z: z * 2.0, lambda z: z, y)
        return softplus(y) + jnp.sum(x) * t["a"]

    g = mp.split_by_frequency(f, X, TREE)
    assert g.n_known > 0 and g.n_unknown > 0
    run = jax.jit(g) if use_jit else g
    out = run(X.astype(jnp.float32), TREE)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out), np.asarray(f(X, TREE)), rtol=1e-6)


def test_split_rejects_unsupported_primitives():
    """scan (also what fori_loop with static bounds lowers to) raises, not miscompiles."""

    def with_scan(x, t):
        return jax.lax.scan(lambda c, xi: (c + xi * t["b"], c), 0.0, x)[1]

    g = mp.split_by_frequency(with_scan, X, TREE)
    with pytest.raises(NotImplementedError, match="scan"):
        g(X.astype(jnp.float32), TREE)


def test_require_x64():
    mp.require_x64()  # the test session enables x64
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="jax_enable_x64"):
        mp.require_x64()
