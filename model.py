from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

Array = jax.Array


def infer_subspace_rank(U: Array, tol: float = 1e-10) -> int:
    U = jnp.asarray(U, dtype=jnp.float32)
    centered = U - jnp.mean(U, axis=0, keepdims=True)
    eigvals = jnp.linalg.eigvalsh(centered.T @ centered)
    return int(jnp.sum(eigvals > tol))


@partial(jax.jit, static_argnames=("rank",))
def build_projector(U: Array, rank: int) -> Array:
    del rank
    centered = U - jnp.mean(U, axis=0, keepdims=True)
    gram = centered @ centered.T
    ridge = 1e-6 * jnp.eye(gram.shape[0], dtype=U.dtype)
    return centered.T @ jnp.linalg.solve(gram + ridge, centered)


@partial(jax.jit, static_argnames=("n",))
def sample_mixture(key: Array, U: Array, n: int) -> Tuple[Array, Array]:
    K, d = U.shape
    key_labels, key_noise = jr.split(key)
    labels = jr.randint(key_labels, (n,), 0, K)
    noise = jr.normal(key_noise, (n, d), dtype=U.dtype)
    samples = U[labels] + noise
    return samples, labels


@jax.jit
def gaussian_logpdf_identity(X: Array, mean: Array) -> Array:
    d = X.shape[-1]
    diff = X - mean
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + jnp.sum(diff * diff, axis=-1))


@jax.jit
def pretraining_responsibilities(X: Array, U: Array) -> Array:
    logits = jax.vmap(lambda u: gaussian_logpdf_identity(X, u))(U).T
    logits = logits - jnp.log(U.shape[0])
    return jax.nn.softmax(logits, axis=1)


@jax.jit
def pretraining_score_batch(X: Array, U: Array) -> Array:
    resp = pretraining_responsibilities(X, U)
    residuals = X[:, None, :] - U[None, :, :]
    return (resp[:, :, None] * residuals).reshape(X.shape[0], -1)


@partial(jax.jit, static_argnames=("n_mc",))
def estimate_fisher_information(U: Array, key: Array, n_mc: int) -> Array:
    X, _ = sample_mixture(key, U, n_mc)
    scores = pretraining_score_batch(X, U)
    return scores.T @ scores / n_mc


@partial(jax.jit, static_argnames=("rank",))
def responsibilities_batch(X: Array, U: Array, rank: int) -> Array:
    P = build_projector(U, rank)
    return responsibilities_batch_from_projector(X, U, P)


@jax.jit
def responsibilities_batch_from_projector(X: Array, U: Array, P: Array) -> Array:
    projected_x = X @ P.T
    projected_u = U @ P.T
    logits = projected_x @ projected_u.T - 0.5 * jnp.sum(projected_u * projected_u, axis=1)
    return jax.nn.softmax(logits, axis=1)


@partial(jax.jit, static_argnames=("rank",))
def feature_map_batch(X: Array, U: Array, rank: int) -> Array:
    P = build_projector(U, rank)
    return feature_map_batch_from_projector(X, U, P)


@jax.jit
def feature_map_batch_from_projector(X: Array, U: Array, P: Array) -> Array:
    pi = responsibilities_batch_from_projector(X, U, P)
    residuals = X[:, None, :] - U[None, :, :]
    projected_residuals = residuals @ P.T
    blocks = jnp.concatenate([pi[:, :, None] * projected_residuals, pi[:, :, None]], axis=2)
    return blocks.reshape(X.shape[0], -1)


@partial(jax.jit, static_argnames=("rank",))
def feature_map_single(x: Array, U: Array, rank: int) -> Array:
    return feature_map_batch(x[None, :], U, rank)[0]


@partial(jax.jit, static_argnames=("rank",))
def predict_batch(X: Array, U: Array, theta: Array, rank: int) -> Array:
    return feature_map_batch(X, U, rank) @ theta


@partial(jax.jit, static_argnames=("rank",))
def fit_population_projection(
    U: Array,
    X_mc: Array,
    f_vals: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Tuple[Array, Array, Array, Array]:
    P = build_projector(U, rank)
    Phi = feature_map_batch_from_projector(X_mc, U, P)
    beta_hat, sigma_hat, c_hat = solve_projection_coefficients(Phi, f_vals, ridge)
    proj_vals = Phi @ beta_hat
    return beta_hat, proj_vals, sigma_hat, c_hat


@jax.jit
def solve_projection_coefficients(
    Phi: Array,
    f_vals: Array,
    ridge: float = 1e-6,
) -> Tuple[Array, Array, Array]:
    n, p = Phi.shape
    sigma_hat = Phi.T @ Phi / n
    c_hat = Phi.T @ f_vals / n
    beta_hat = jnp.linalg.solve(sigma_hat + ridge * jnp.eye(p, dtype=Phi.dtype), c_hat)
    return beta_hat, sigma_hat, c_hat


@partial(jax.jit, static_argnames=("rank",))
def projected_values_batch(
    U: Array,
    X_eval: Array,
    X_mc: Array,
    f_vals: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Array:
    P = build_projector(U, rank)
    Phi_proj = feature_map_batch_from_projector(X_mc, U, P)
    beta_hat, _, _ = solve_projection_coefficients(Phi_proj, f_vals, ridge)
    Phi_eval = feature_map_batch_from_projector(X_eval, U, P)
    return Phi_eval @ beta_hat


@partial(jax.jit, static_argnames=("rank",))
def local_sensitivity_batch(
    U_star: Array,
    X_eval: Array,
    X_proj: Array,
    f_vals: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Array:
    jacobian = jax.jacrev(projected_values_batch, argnums=0)(
        U_star, X_eval, X_proj, f_vals, rank, ridge
    )
    return -jnp.moveaxis(jacobian.reshape(U_star.shape + (X_eval.shape[0],)), -1, 0).reshape(
        X_eval.shape[0], -1
    )


@jax.jit
def regularized_inverse(matrix: Array, ridge: float = 1e-6) -> Array:
    dim = matrix.shape[0]
    ridge_eye = ridge * jnp.eye(dim, dtype=matrix.dtype)
    return jnp.linalg.pinv(matrix + ridge_eye)


@partial(jax.jit, static_argnames=("rank",))
def quadratic_risk_from_sigma(
    U_star: Array,
    sigma_z: Array,
    X_proj: Array,
    f_vals: Array,
    X_eval: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Array:
    sensitivities = local_sensitivity_batch(U_star, X_eval, X_proj, f_vals, rank, ridge)
    quadratic_terms = jnp.einsum("ni,ij,nj->n", sensitivities, sigma_z, sensitivities)
    return jnp.mean(quadratic_terms)


def estimate_expected_l2_sq(
    U_star: Array,
    theta_star: Array,
    key: Array,
    n_fisher_mc: int,
    n_proj_mc: int,
    n_eval_mc: int,
    rank: int | None = None,
    fisher_ridge: float = 1e-6,
    proj_ridge: float = 1e-6,
) -> tuple[Array, dict[str, Array]]:
    U_star = jnp.asarray(U_star, dtype=jnp.float32)
    theta_star = jnp.asarray(theta_star, dtype=jnp.float32)
    rank = infer_subspace_rank(U_star) if rank is None else rank

    key_fisher, key_proj, key_eval = jr.split(key, 3)

    fisher_hat = estimate_fisher_information(U_star, key_fisher, n_fisher_mc)
    sigma_z = regularized_inverse(fisher_hat, fisher_ridge)

    X_proj, _ = sample_mixture(key_proj, U_star, n_proj_mc)
    f_vals = predict_batch(X_proj, U_star, theta_star, rank)

    X_eval, _ = sample_mixture(key_eval, U_star, n_eval_mc)
    risk = quadratic_risk_from_sigma(U_star, sigma_z, X_proj, f_vals, X_eval, rank, proj_ridge)

    diagnostics = {
        "fisher_hat": fisher_hat,
        "sigma_z": sigma_z,
        "X_proj": X_proj,
        "X_eval": X_eval,
        "f_vals": f_vals,
    }
    return risk, diagnostics
