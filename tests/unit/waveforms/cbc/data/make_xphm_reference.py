"""Regenerate xphm_reference.npz: outputs of the base IMRPhenomXPHM implementations.

Run from the repository root (needs the git history and CPU only):

    python tests/unit/waveforms/cbc/data/make_xphm_reference.py

Stored arrays
-------------
theta            [11, 11] quick batch (7 box draws, row 4 with irregular angles at
                          256 nodes; 2 edge cases; 2 MSA_ERROR draws)
freqs            [97]     quick grid (multibanded, every 14th point)
base_hp/base_hc  2c8d63b, float64 (before the optimisations, no NNLO fallback)
nnlo_hp/nnlo_hc  24fc1a1, float64 (NNLO fallback on, not optimised)
base32_hp/hc     2c8d63b, float32 (process without x64)
grad_freqs       [25]     tiny grid for the gradient reference
base_grad        [4]      d/d(m1, m2, chi1z, distance) of 1e40 * sum(|hp|^2 + |hc|^2), row 0, 2c8d63b
irregularity     [11]     angle_node_irregularity at 256 nodes (current tree) on the quick grid
msa_error        [11]     LAL MSA_ERROR flag (current tree)
"""

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "xphm_reference.npz"
sys.path.insert(0, str(ROOT))

from tests.helpers.xphm import quick_grid, sample_bbh_box

# Edge cases and selected draws (generate_xphm argument order).
EDGE = [
    # equal masses, equal spins (symmetric)
    [30.0, 30.0, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 1000.0, 1.0, 0.5],
    # aligned spins only (no in-plane component)
    [45.0, 15.0, 0.0, 0.0, 0.6, 0.0, 0.0, -0.4, 2000.0, 0.3, 2.0],
]
# MSA_ERROR draws (msa_error_draws.json entries 3 and 6), irregular at 256 nodes


def quick_theta() -> np.ndarray:
    import json

    d = json.loads(
        (Path(__file__).resolve().parent.parent / "msa_error_draws.json").read_text()
    )
    msa = [d["msa_error"][3], d["msa_error"][6]]
    return np.concatenate([sample_bbh_box(7, seed=11), np.array(EDGE), np.array(msa)])


GEN = r"""
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", sys.argv[2] == "64")
import jax.numpy as jnp
import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as X
z = np.load(sys.argv[1]); dt = jnp.float64 if sys.argv[2] == "64" else jnp.float32
th = jnp.asarray(z["theta"], dt); f = jnp.asarray(z["freqs"], dt)
hp, hc = jax.jit(jax.vmap(lambda t: X.generate_xphm(*[t[i] for i in range(11)], f, 20.0)))(th)
out = dict(hp=np.asarray(hp), hc=np.asarray(hc))
if sys.argv[2] == "64" and sys.argv[4] == "grad":
    g = jnp.asarray(z["grad_freqs"]); t = th[0]
    def loss(m1, m2, c1z, d):
        p = [m1, m2, t[2], t[3], c1z, t[5], t[6], t[7], d, t[9], t[10]]
        hp, hc = X.generate_xphm(*p, g, 20.0)
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2) * 1e40
    out["grad"] = np.array(jax.jit(jax.jacfwd(loss, argnums=(0, 1, 2, 3)))(t[0], t[1], t[4], t[8]))
np.savez(sys.argv[3], **out)
"""


def run_commit(commit: str, bits: str, inputs: Path, grad: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "src.tar"
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "-o", str(tar_path), commit, "src"],
            check=True,
        )
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        out = Path(tmp) / "out.npz"
        env = {
            "PYTHONPATH": str(Path(tmp) / "src"),
            "JAX_PLATFORMS": "cpu",
            "PATH": "/usr/bin:/bin",
        }
        subprocess.run(
            [
                sys.executable,
                "-c",
                GEN,
                str(inputs),
                bits,
                str(out),
                "grad" if grad else "",
            ],
            check=True,
            env=env,
        )
        with np.load(out) as z:
            return {k: z[k] for k in z.files}


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomX_NNLO as pNNLO
    import ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM as X
    import ripplegw.waveforms.cbc.IMRPhenomX.LALSimIMRPhenomX_precession as pPrec
    from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
        IMRPhenomX_Initialize_MSA_System,
        lal_M_sec,
    )

    theta, freqs = quick_theta(), quick_grid()
    grad_freqs = freqs[::4]
    res = {"theta": theta, "freqs": freqs, "grad_freqs": grad_freqs}
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "in.npz"
        np.savez(inputs, theta=theta, freqs=freqs, grad_freqs=grad_freqs)
        from concurrent.futures import ThreadPoolExecutor

        jobs = [
            ("2c8d63b", "64", True),
            ("24fc1a1", "64", False),
            ("2c8d63b", "32", False),
        ]
        with ThreadPoolExecutor(len(jobs)) as pool:
            b, n, b32 = pool.map(
                lambda j: run_commit(j[0], j[1], inputs, grad=j[2]), jobs
            )
        res.update(base_hp=b["hp"], base_hc=b["hc"], base_grad=b["grad"])
        res.update(nnlo_hp=n["hp"], nnlo_hc=n["hc"])
        res.update(base32_hp=b32["hp"], base32_hc=b32["hc"])

    def flags(t):
        _, _, setup, _ = X.precession_frame(
            *[t[i] for i in range(8)], t[10], t[9], 20.0
        )
        mf = jnp.asarray(freqs) * lal_M_sec(t[0], t[1])
        irr = pPrec.angle_node_irregularity(mf, (1, 2, 3, 4), setup, 256)
        init = IMRPhenomX_Initialize_MSA_System(
            *[t[i] for i in range(8)], reference_frequency=20.0
        )
        return irr, pNNLO.msa_error_from_init(init)

    if not hasattr(pPrec, "angle_node_irregularity"):  # tree without interpolation
        raise SystemExit("needs a tree with angle_node_irregularity (flags)")
    irr, err = jax.jit(jax.vmap(flags))(jnp.asarray(theta))
    res.update(irregularity=np.asarray(irr), msa_error=np.asarray(err))
    np.savez_compressed(OUT, **res)
    print(
        "wrote",
        OUT,
        "irregularity",
        np.round(res["irregularity"], 3),
        "msa_error",
        res["msa_error"],
    )


if __name__ == "__main__":
    main()
