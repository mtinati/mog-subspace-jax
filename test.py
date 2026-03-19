import jax
import jax.numpy as jnp

from model import (
    estimate_expected_l2_sq,
    estimate_fisher_information,
    feature_map_batch,
    infer_subspace_rank,
    local_sensitivity_batch,
    predict_batch,
    regularized_inverse,
    sample_mixture,
)


def make_U_star(K: int, d: int, beta: float) -> jax.Array:
    if d < K:
        raise ValueError("Need d >= K.")
    return beta * jnp.eye(K, d, dtype=jnp.float32)


def main() -> None:
    K = 3
    d = 6
    beta = 1.5
    n_fisher_mc = 512
    n_proj_mc = 256
    n_eval_mc = 128

    U_star = make_U_star(K, d, beta)
    rank = infer_subspace_rank(U_star)
    theta_star = jnp.arange(1, K * (d + 1) + 1, dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    key_fisher, key_proj, key_eval, key_full = jax.random.split(key, 4)

    X_proj, _ = sample_mixture(key_proj, U_star, n_proj_mc)
    X_eval, _ = sample_mixture(key_eval, U_star, n_eval_mc)
    f_vals = predict_batch(X_proj, U_star, theta_star, rank)
    fisher_hat = estimate_fisher_information(U_star, key_fisher, n_fisher_mc)
    sigma_z = regularized_inverse(fisher_hat, ridge=1e-6)
    Phi = feature_map_batch(X_proj, U_star, rank)
    sensitivities = local_sensitivity_batch(U_star, X_eval, X_proj, f_vals, rank, ridge=1e-6)
    risk, diagnostics = estimate_expected_l2_sq(
        U_star=U_star,
        theta_star=theta_star,
        key=key_full,
        n_fisher_mc=n_fisher_mc,
        n_proj_mc=n_proj_mc,
        n_eval_mc=n_eval_mc,
        rank=rank,
        fisher_ridge=1e-6,
        proj_ridge=1e-6,
    )

    print("rank =", rank)
    print("feature matrix shape =", Phi.shape)
    print("fisher shape =", fisher_hat.shape)
    print("sigma_z trace =", float(jnp.trace(sigma_z)))
    print("sensitivity batch shape =", sensitivities.shape)
    print("estimated risk =", float(risk))
    print("diagnostic keys =", sorted(diagnostics.keys()))


if __name__ == "__main__":
    main()
