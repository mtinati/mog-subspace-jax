from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from model import estimate_expected_l2_sq


def make_U_star(K: int, d: int, beta: float) -> jax.Array:
    if d < K:
        raise ValueError(f"Need d >= K, but got d={d}, K={K}.")
    return beta * jnp.eye(K, d, dtype=jnp.float32)


def make_theta_star(K: int, d: int) -> jax.Array:
    blocks = []
    for i in range(K):
        weight = 10 / float(i + 1)
        block = weight * jnp.ones((d + 1,), dtype=jnp.float32)
        blocks.append(block)
    theta = jnp.concatenate(blocks)
    return 10 * theta / jnp.linalg.norm(theta)


def estimate_expected_L2_sq(
    K: int,
    d: int,
    beta: float,
    n_fisher_mc: int = 200000,
    n_proj_mc: int = 40000,
    n_eval_mc: int = 40000,
    fisher_ridge: float = 1e-6,
    proj_ridge: float = 1e-6,
    seed: int = 0,
    eval_chunk_size: int | None = 256,
    proj_chunk_size: int | None = None,
    rank: int | None = None,
    include_beta_jacobian: bool = True,
) -> float:
    U_star = make_U_star(K=K, d=d, beta=beta)
    theta_star = make_theta_star(K=K, d=d)
    key = jax.random.PRNGKey(seed)

    risk, _ = estimate_expected_l2_sq(
        U_star=U_star,
        theta_star=theta_star,
        key=key,
        n_fisher_mc=n_fisher_mc,
        n_proj_mc=n_proj_mc,
        n_eval_mc=n_eval_mc,
        rank=rank,
        fisher_ridge=fisher_ridge,
        proj_ridge=proj_ridge,
        eval_chunk_size=eval_chunk_size,
        proj_chunk_size=proj_chunk_size,
        include_beta_jacobian=include_beta_jacobian,
    )
    risk_value = float(risk)
    del risk
    return risk_value


def release_memory() -> None:
    gc.collect()


def run_point_and_cleanup(K: int, d: int, beta: float, **kwargs) -> float:
    try:
        return estimate_expected_L2_sq(K=K, d=d, beta=beta, **kwargs)
    finally:
        release_memory()


def sweep_K(K_values, d_fixed: int, beta_fixed: float, **kwargs) -> np.ndarray:
    results = []
    for K in K_values:
        if d_fixed < K:
            raise ValueError(f"For K sweep, need d_fixed >= max(K_values). Failed at K={K}.")
        val = run_point_and_cleanup(K=K, d=d_fixed, beta=beta_fixed, **kwargs)
        results.append(val)
        print(f"K={K}, d={d_fixed}, beta={beta_fixed} -> E||L(Z)||^2 ≈ {val:.6f}")
    return np.asarray(results, dtype=float)


def sweep_d(d_values, K_fixed: int, beta_fixed: float, **kwargs) -> np.ndarray:
    rank_fixed = K_fixed - 1
    results = []
    for d in d_values:
        if d < K_fixed:
            raise ValueError(f"For d sweep, need every d >= K_fixed={K_fixed}. Failed at d={d}.")
        val = run_point_and_cleanup(K=K_fixed, d=d, beta=beta_fixed, rank=rank_fixed, **kwargs)
        results.append(val)
        print(f"K={K_fixed}, d={d}, beta={beta_fixed} -> E||L(Z)||^2 ≈ {val:.6f}")
    return np.asarray(results, dtype=float)


def sweep_beta(beta_values, K_fixed: int, d_fixed: int, **kwargs) -> np.ndarray:
    if d_fixed < K_fixed:
        raise ValueError(f"Need d_fixed >= K_fixed, got d_fixed={d_fixed}, K_fixed={K_fixed}.")
    rank_fixed = K_fixed - 1
    results = []
    for beta in beta_values:
        val = run_point_and_cleanup(K=K_fixed, d=d_fixed, beta=beta, rank=rank_fixed, **kwargs)
        results.append(val)
        print(f"K={K_fixed}, d={d_fixed}, beta={beta} -> E||L(Z)||^2 ≈ {val:.6f}")
    return np.asarray(results, dtype=float)


def default_output_path(prefix: str = "mog_three_sweeps") -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("saved_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_{timestamp}.npz"


def save_results(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_payload = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            arrays_payload[key] = value
        elif isinstance(value, (list, tuple)):
            arrays_payload[key] = np.asarray(value)
        elif isinstance(value, dict):
            arrays_payload[key] = np.asarray(json.dumps(value))
        else:
            arrays_payload[key] = np.asarray(value)
    np.savez(output_path, **arrays_payload)


def load_results(output_path: Path) -> dict[str, object]:
    with np.load(output_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    metadata = json.loads(str(payload["metadata"].item()))
    payload["metadata"] = metadata
    return payload


def compute_three_sweeps(
    *,
    K0: int = 4,
    d0: int = 40,
    beta0: float = 2.0,
    K_values: np.ndarray | None = None,
    d_values: np.ndarray | None = None,
    beta_values: np.ndarray | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    **common_kwargs,
) -> tuple[dict[str, object], Path]:
    output = Path(output_path) if output_path is not None else default_output_path()
    if output.exists() and not overwrite:
        print(f"Loading saved results from {output}")
        return load_results(output), output

    K_values = np.asarray(np.arange(2, 15, 1) if K_values is None else K_values)
    d_values = np.asarray(
        np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(1000), 30))).astype(int))
        if d_values is None
        else d_values
    )
    beta_values = np.asarray(
        np.geomspace(0.5, 10.0, 20) if beta_values is None else beta_values
    )

    y_K = sweep_K(K_values=K_values, d_fixed=d0, beta_fixed=beta0, **common_kwargs)
    y_d = sweep_d(d_values=d_values, K_fixed=K0, beta_fixed=beta0, **common_kwargs)
    y_beta = sweep_beta(beta_values=beta_values, K_fixed=K0, d_fixed=d0, **common_kwargs)

    metadata = {
        "K0": int(K0),
        "d0": int(d0),
        "beta0": float(beta0),
        "common_kwargs": common_kwargs,
    }

    payload = {
        "K_values": K_values,
        "y_K": y_K,
        "d_values": d_values,
        "y_d": y_d,
        "beta_values": beta_values,
        "y_beta": y_beta,
        "metadata": metadata,
    }
    save_results(output, payload)
    print(f"Saved results to {output}")
    return payload, output


def plot_three_sweeps(results: dict[str, object], figure_path: str | Path | None = None) -> Path | None:
    import matplotlib.pyplot as plt

    metadata = results["metadata"]
    K0 = metadata["K0"]
    d0 = metadata["d0"]     
    beta0 = metadata["beta0"]

    K_values = np.asarray(results["K_values"])
    y_K = np.asarray(results["y_K"])
    d_values = np.asarray(results["d_values"])
    y_d = np.asarray(results["y_d"])
    beta_values = np.asarray(results["beta_values"])
    y_beta = np.asarray(results["y_beta"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(K_values, y_K, marker="o")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel(r"$\mathbb{E}\|L(Z)\|_{L^2(\mu_{\mathrm{down}})}^2$")
    axes[0].set_title(rf"(fixed $d={d0}$, $\beta={beta0}$)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(d_values, y_d, marker="o")
    axes[1].set_xlabel("d")
    axes[1].set_ylabel(r"$\mathbb{E}\|L(Z)\|_{L^2(\mu_{\mathrm{down}})}^2$")
    axes[1].set_title(rf"(fixed $K={K0}$, $\beta={beta0}$)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(beta_values, y_beta, marker="o")
    axes[2].set_xlabel(r"$\beta$")
    axes[2].set_ylabel(r"$\mathbb{E}\|L(Z)\|_{L^2(\mu_{\mathrm{down}})}^2$")
    axes[2].set_title(rf"(fixed $K={K0}$, $d={d0}$)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if figure_path is None:
        plt.show()
        return None

    figure_output = Path(figure_path)
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {figure_output}")
    return figure_output


def make_three_plots() -> None:
    common_kwargs = dict(
        n_fisher_mc=40000,
        n_proj_mc=100000,
        n_eval_mc=100000,
        fisher_ridge=1e-6,
        proj_ridge=1e-6,
        seed=int(time.time()),
        eval_chunk_size=512,
        proj_chunk_size=8192,
        include_beta_jacobian=True,
    )

    results, output_path = compute_three_sweeps(
        K0=4,
        d0=20,
        beta0=2.0,
        **common_kwargs,
    )

    figure_path = output_path.with_suffix(".png")
    plot_three_sweeps(results, figure_path=figure_path)


if __name__ == "__main__":
    make_three_plots()
