import jax
import jax.numpy as jnp
import time

from model import estimate_expected_l2_sq


def make_U_star(K: int, d: int, beta: float) -> jax.Array:
    if d < K:
        raise ValueError(f"Need d >= K, but got d={d}, K={K}.")
    return beta * jnp.eye(K, d, dtype=jnp.float32)


def make_theta_star(K: int, d: int) -> jax.Array:
    blocks = []
    for i in range(K):
        weight = ((-1.0) ** i) / float(i + 1)
        block = weight * jnp.ones((d + 1,), dtype=jnp.float32)
        blocks.append(block)
    theta = jnp.concatenate(blocks)
    return theta / jnp.linalg.norm(theta)


def estimate_expected_L2_sq(
    K: int,
    d: int,
    beta: float,
    n_fisher_mc: int = 200000,
    n_proj_mc: int = 4000,
    n_eval_mc: int = 4000,
    fisher_ridge: float = 1e-6,
    proj_ridge: float = 1e-6,
    seed: int = 0,
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
        fisher_ridge=fisher_ridge,
        proj_ridge=proj_ridge,
    )
    return float(risk)


def sweep_K(K_values, d_fixed: int, beta_fixed: float, **kwargs) -> jax.Array:
    results = []
    for K in K_values:
        if d_fixed < K:
            raise ValueError(f"For K sweep, need d_fixed >= max(K_values). Failed at K={K}.")
        val = estimate_expected_L2_sq(K=K, d=d_fixed, beta=beta_fixed, **kwargs)
        results.append(val)
        print(f"K={K}, d={d_fixed}, beta={beta_fixed} -> E||L(Z)||^2 ≈ {val:.6f}")
    return jnp.array(results)


def sweep_d(d_values, K_fixed: int, beta_fixed: float, **kwargs) -> jax.Array:
    results = []
    for d in d_values:
        if d < K_fixed:
            raise ValueError(f"For d sweep, need every d >= K_fixed={K_fixed}. Failed at d={d}.")
        val = estimate_expected_L2_sq(K=K_fixed, d=d, beta=beta_fixed, **kwargs)
        results.append(val)
        print(f"K={K_fixed}, d={d}, beta={beta_fixed} -> E||L(Z)||^2 ≈ {val:.6f}")
    return jnp.array(results)


def sweep_beta(beta_values, K_fixed: int, d_fixed: int, **kwargs) -> jax.Array:
    if d_fixed < K_fixed:
        raise ValueError(f"Need d_fixed >= K_fixed, got d_fixed={d_fixed}, K_fixed={K_fixed}.")
    results = []
    for beta in beta_values:
        val = estimate_expected_L2_sq(K=K_fixed, d=d_fixed, beta=beta, **kwargs)
        results.append(val)
        print(f"K={K_fixed}, d={d_fixed}, beta={beta} -> E||L(Z)||^2 ≈ {val:.6f}")
    return jnp.array(results)


def make_three_plots() -> None:
    import matplotlib.pyplot as plt

    K0 = 4
    d0 = 10
    beta0 = 2.0

    common_kwargs = dict(
        n_fisher_mc=10000,
        n_proj_mc=1000,
        n_eval_mc=10000,
        fisher_ridge=1e-6,
        proj_ridge=1e-6,
        seed=int(time.time()),
    )

    K_values = jnp.arange(10, 10, 1)
    y_K = sweep_K(K_values=K_values, d_fixed=d0, beta_fixed=beta0, **common_kwargs)

    d_values = jnp.arange(5, 50, 2)
    y_d = sweep_d(d_values=d_values, K_fixed=K0, beta_fixed=beta0, **common_kwargs)

    beta_values = jnp.arange(5, 5.0, 0.5)
    y_beta = sweep_beta(beta_values=beta_values, K_fixed=K0, d_fixed=d0, **common_kwargs)

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
    plt.savefig("mog_three_parameter_sweeps.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    make_three_plots()
