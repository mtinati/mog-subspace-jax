# mog-subspace-jax

JAX code for Monte Carlo experiments in a mixture-of-Gaussians subspace model. The repository estimates an expected `L^2` error quantity and runs parameter sweeps over the number of components, ambient dimension, and separation parameter.

## Files

- `model.py`: core sampling, projection, Fisher-information, sensitivity, and risk-estimation routines
- `main.py`: experiment driver for running sweeps and saving plots/results

## Requirements

- Python 3
- JAX
- NumPy
- Matplotlib

## Run

```bash
python main.py
```

This runs the default sweeps and saves results in `saved_results/`.

## Changing Parameters

The main experiment settings are in `main.py`, inside `make_three_plots()`. To change the Monte Carlo sample sizes, edit:

- `n_fisher_mc`
- `n_proj_mc`
- `n_eval_mc`

You can also change:

- `K0`, `d0`, `beta0` for the baseline sweep settings
- `eval_chunk_size` and `proj_chunk_size` for memory/performance tradeoffs

The sweep grids are defined in `compute_three_sweeps()` through:

- `K_values`
- `d_values`
- `beta_values`
