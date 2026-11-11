# FAQ

## Float precision

JAX defaults to float32, but gravitational-wave waveform computations typically require float64 precision.
Always enable it at the top of your script, before any JAX operations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Without this, you may see unexpected numerical errors or inaccurate waveforms, particularly at high frequencies or for long signals.

## JIT compilation time

The first call to a JIT-compiled waveform (e.g. via `jax.jit`) triggers XLA compilation, which can take several seconds.
This is normal — subsequent calls will be much faster.
If you are timing ripple for benchmarking purposes, discard the first call.

To disable JIT for debugging:

```python
jax.config.update("jax_disable_jit", True)
```

## Compilation is slow for complex models

If you wrap a ripple waveform inside a larger likelihood with many operations or Python-level loops, JAX may take a long time to compile the full computational graph.
Replacing Python loops with `jax.lax.scan` or `jax.vmap` where possible can significantly reduce compilation time.

## `ripplegw.IMRPhenomD` (or `waveform_preset`) raises `AttributeError`

Older ripple code constructed models directly off the top-level module (`ripplegw.IMRPhenomD(f_ref=20.0)`) or looked them up in a `waveform_preset` dict.
Neither exists anymore.
The only construction path is the registry factory:

```python
waveform = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
```

See [Working with Waveforms](guides/waveforms.md) for the full interface and `ripplegw.list_waveforms()` for every registered name.

## Where is `t_c` (time of coalescence)?

It isn't exposed.
Every built-in model fixes the time of coalescence internally and only exposes `phase_c` (coalescence phase) and `iota` (inclination) as extrinsic parameters — see [Parameters and Conventions](notes/parameters.md).
If your use case needs to vary `t_c`, you currently need to apply the standard linear-in-frequency phase shift ($e^{-2\pi i f\,\delta t_c}$) yourself on the returned strain.

## Running ripple next to PyTorch on one GPU

- Before JAX initialises, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and cap JAX's share with `XLA_CLIENT_MEM_FRACTION` (e.g. `0.3`; jax ≥ 0.12 name, do not also set its deprecated alias `XLA_PYTHON_CLIENT_MEM_FRACTION`). Otherwise JAX may grow to 75 % of the GPU memory and starve PyTorch.
- float64 and `precision="mixed"` (IMRPhenomXPHM) need `jax.config.update("jax_enable_x64", True)` before any array is created.
- To hand waveforms to PyTorch without a copy, use `torch.from_dlpack(jax_array)`. It is zero-copy and synchronises the CUDA streams.

## Faster IMRPhenomXPHM: mixed precision and interpolated angles

`generate_xphm` and `IMRPhenomXPHM(...)` take `precision="float64" | "mixed" | "float32"`. The default, `"float64"`, is exact. `"mixed"` runs the per-binary setup and the precession angles in float64 and the per-frequency modes and twist-up in float32, and returns complex64. It needs `jax_enable_x64`.

For batches, use `IMRPhenomXPHM_stages.generate_xphm_batch(params, frequency_array, reference_frequency, ...)`, with `params` of shape `[batch, 11]` in the argument order of `generate_xphm` (m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, distance [Mpc], inclination, phi0). Its defaults are exact as well. Interpolated precession angles (`angle_interp_nodes`) are offered only there. Binaries whose angles are too irregular to interpolate are recomputed exactly; if more than `ceil(angle_interp_fallback_max_fraction * batch)` are flagged, the whole batch is recomputed exactly. The function returns `(hp, hc, n_flagged, overflow)`. Make every argument except `params` and `frequency_array` static (e.g. `functools.partial` before `jax.jit`).

The fast setting, `precision="mixed", angle_interp_nodes=256` (fallback threshold 1 rad, at most 5 % of a batch recomputed individually), was tuned and validated for batched generation (batches of 1024–16384) on one BBH box and grid:

- box: chirp mass 15–150 Msun (log-uniform), component masses 10–120 Msun, q ≥ 1/8, spin magnitudes ≤ 0.99 with isotropic directions, f_ref = 20 Hz;
- grid: 20–1024 Hz, either uniform at 0.125 Hz or multibanded with 1358 points.

The worst mismatch against float64 (aLIGO PSD) was 8.4e-5 over 12288 draws. That is a sample maximum, not a bound; p99 was 9e-8.

On an A100 at batch 4096 on the multibanded grid, the fast setting runs at 336k waveforms/s. That is 2.7× the fork's exact float64 path (124.5k/s) and 6.3× the float64 path before these changes (52.9k/s).

Other uses need re-validation before relying on these numbers: other mass or spin ranges, a lower f_min, other grids, small batches, single waveforms or gradients. Also note that float32 strain (~1e-21 to 1e-29) underflows when squared, so whiten or rescale before forming inner products in float32.
