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


@jax.jit
def solve_projection_from_moments(
    sum_phi_outer: Array,
    sum_phi_f: Array,
    n_total: int,
    ridge: float = 1e-6,
) -> Tuple[Array, Array, Array]:
    sigma_hat = sum_phi_outer / n_total
    c_hat = sum_phi_f / n_total
    p = sigma_hat.shape[0]
    beta_hat = jnp.linalg.solve(sigma_hat + ridge * jnp.eye(p, dtype=sigma_hat.dtype), c_hat)
    return beta_hat, sigma_hat, c_hat


@partial(jax.jit, static_argnames=("rank",))
def beta_from_projection_samples(
    U: Array,
    X_mc: Array,
    f_vals: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Array:
    P = build_projector(U, rank)
    Phi_proj = feature_map_batch_from_projector(X_mc, U, P)
    beta_hat, _, _ = solve_projection_coefficients(Phi_proj, f_vals, ridge)
    return beta_hat


@partial(jax.jit, static_argnames=("rank",))
def moments_from_projection_samples(
    U: Array,
    X_mc: Array,
    f_vals: Array,
    rank: int,
) -> Tuple[Array, Array]:
    P = build_projector(U, rank)
    Phi_proj = feature_map_batch_from_projector(X_mc, U, P)
    n = X_mc.shape[0]
    sigma_hat = Phi_proj.T @ Phi_proj / n
    c_hat = Phi_proj.T @ f_vals / n
    return sigma_hat, c_hat


@partial(jax.jit, static_argnames=("rank",))
def beta_and_jacobian_ift_samples(
    U: Array,
    X_mc: Array,
    f_vals: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Tuple[Array, Array]:
    sigma_hat, c_hat = moments_from_projection_samples(U, X_mc, f_vals, rank)
    p = sigma_hat.shape[0]
    sigma_reg = sigma_hat + ridge * jnp.eye(p, dtype=U.dtype)
    beta_hat = jnp.linalg.solve(sigma_reg, c_hat)

    def flattened_moments(U_: Array) -> Array:
        sigma_val, c_val = moments_from_projection_samples(U_, X_mc, f_vals, rank)
        return jnp.concatenate([sigma_val.reshape(-1), c_val], axis=0)

    jac_flat = jax.jacobian(flattened_moments)(U)
    jac_sigma = jac_flat[: p * p].reshape(p, p, *U.shape)
    jac_c = jac_flat[p * p :].reshape(p, *U.shape)
    rhs = jac_c - jnp.einsum("abij,b->aij", jac_sigma, beta_hat)
    jac_beta_flat = jnp.linalg.solve(sigma_reg, rhs.reshape(p, -1))
    jac_beta = jac_beta_flat.reshape((p,) + U.shape)
    return beta_hat, jac_beta


@partial(jax.jit, static_argnames=("rank", "n_proj", "chunk_size"))
def beta_from_projection_streaming(
    U: Array,
    U_data: Array,
    theta_star: Array,
    key_proj: Array,
    n_proj: int,
    rank: int,
    chunk_size: int,
    ridge: float = 1e-6,
) -> Array:
    n_chunks = (n_proj + chunk_size - 1) // chunk_size
    p = U.shape[0] * (U.shape[1] + 1)
    P = build_projector(U, rank)
    P_data = build_projector(U_data, rank)

    def body_fun(idx: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        sum_phi_outer, sum_phi_f = carry
        chunk_key = jr.fold_in(key_proj, idx)
        X_chunk, _ = sample_mixture(chunk_key, U_data, chunk_size)
        f_chunk = feature_map_batch_from_projector(X_chunk, U_data, P_data) @ theta_star

        Phi_chunk = feature_map_batch_from_projector(X_chunk, U, P)

        remaining = n_proj - idx * chunk_size
        valid_count = jnp.minimum(chunk_size, remaining)
        mask = (jnp.arange(chunk_size) < valid_count).astype(U.dtype)

        weighted_phi = Phi_chunk * mask[:, None]
        weighted_f = f_chunk * mask

        sum_phi_outer = sum_phi_outer + weighted_phi.T @ Phi_chunk
        sum_phi_f = sum_phi_f + weighted_phi.T @ weighted_f
        return sum_phi_outer, sum_phi_f

    init_outer = jnp.zeros((p, p), dtype=U.dtype)
    init_cross = jnp.zeros((p,), dtype=U.dtype)
    sum_phi_outer, sum_phi_f = jax.lax.fori_loop(
        0,
        n_chunks,
        body_fun,
        (init_outer, init_cross),
    )
    beta_hat, _, _ = solve_projection_from_moments(sum_phi_outer, sum_phi_f, n_proj, ridge)
    return beta_hat


@partial(jax.jit, static_argnames=("rank",))
def projected_values_with_fixed_beta(
    U: Array,
    X_eval: Array,
    beta: Array,
    rank: int,
) -> Array:
    Phi_eval = feature_map_batch(X_eval, U, rank)
    return Phi_eval @ beta


@partial(jax.jit, static_argnames=("rank",))
def projected_values_batch(
    U: Array,
    X_eval: Array,
    X_mc: Array,
    f_vals: Array,
    rank: int,
    ridge: float = 1e-6,
) -> Array:
    beta_hat = beta_from_projection_samples(U, X_mc, f_vals, rank, ridge)
    return projected_values_with_fixed_beta(U, X_eval, beta_hat, rank)


@partial(jax.jit, static_argnames=("rank", "include_beta_jacobian"))
def local_sensitivity_batch(
    U_star: Array,
    X_eval: Array,
    X_proj: Array,
    f_vals: Array,
    rank: int,
    include_beta_jacobian: bool = True,
    ridge: float = 1e-6,
) -> Array:
    if not include_beta_jacobian:
        beta_star = beta_from_projection_samples(U_star, X_proj, f_vals, rank, ridge)
        jac_phi_beta = jax.jacrev(projected_values_with_fixed_beta, argnums=0)(
            U_star, X_eval, beta_star, rank
        )
        term_phi = jac_phi_beta.reshape(X_eval.shape[0], -1)
        return -term_phi

    beta_star, jac_beta = beta_and_jacobian_ift_samples(U_star, X_proj, f_vals, rank, ridge)
    jac_phi_beta = jax.jacrev(projected_values_with_fixed_beta, argnums=0)(
        U_star, X_eval, beta_star, rank
    )
    term_phi = jac_phi_beta.reshape(X_eval.shape[0], -1)
    phi_eval = feature_map_batch(X_eval, U_star, rank)
    jac_beta_flat = jac_beta.reshape(beta_star.shape[0], -1)
    term_beta = phi_eval @ jac_beta_flat

    return -(term_phi + term_beta)


@partial(jax.jit, static_argnames=("rank", "n_proj", "proj_chunk_size", "include_beta_jacobian"))
def local_sensitivity_batch_streaming_proj(
    U_star: Array,
    X_eval: Array,
    theta_star: Array,
    key_proj: Array,
    n_proj: int,
    rank: int,
    proj_chunk_size: int,
    include_beta_jacobian: bool = True,
    ridge: float = 1e-6,
) -> Array:
    beta_star = beta_from_projection_streaming(
        U_star, U_star, theta_star, key_proj, n_proj, rank, proj_chunk_size, ridge
    )

    jac_phi_beta = jax.jacrev(projected_values_with_fixed_beta, argnums=0)(
        U_star, X_eval, beta_star, rank
    )
    term_phi = jac_phi_beta.reshape(X_eval.shape[0], -1)

    if not include_beta_jacobian:
        return -term_phi

    phi_eval = feature_map_batch(X_eval, U_star, rank)
    jac_beta = jax.jacrev(beta_from_projection_streaming, argnums=0)(
        U_star, U_star, theta_star, key_proj, n_proj, rank, proj_chunk_size, ridge
    )
    jac_beta_flat = jac_beta.reshape(beta_star.shape[0], -1)
    term_beta = phi_eval @ jac_beta_flat

    return -(term_phi + term_beta)


@jax.jit
def regularized_inverse(matrix: Array, ridge: float = 1e-6) -> Array:
    dim = matrix.shape[0]
    ridge_eye = ridge * jnp.eye(dim, dtype=matrix.dtype)
    return jnp.linalg.pinv(matrix + ridge_eye)


@partial(jax.jit, static_argnames=("rank", "include_beta_jacobian"))
def quadratic_risk_from_sigma(
    U_star: Array,
    sigma_z: Array,
    X_proj: Array,
    f_vals: Array,
    X_eval: Array,
    rank: int,
    include_beta_jacobian: bool = True,
    ridge: float = 1e-6,
) -> Array:
    sensitivities = local_sensitivity_batch(
        U_star, X_eval, X_proj, f_vals, rank, include_beta_jacobian, ridge
    )
    quadratic_terms = jnp.einsum("ni,ij,nj->n", sensitivities, sigma_z, sensitivities)
    return jnp.mean(quadratic_terms)


@partial(jax.jit, static_argnames=("rank", "chunk_size", "include_beta_jacobian"))
def quadratic_risk_from_sigma_chunked(
    U_star: Array,
    sigma_z: Array,
    X_proj: Array,
    f_vals: Array,
    X_eval: Array,
    rank: int,
    chunk_size: int,
    include_beta_jacobian: bool = True,
    ridge: float = 1e-6,
) -> Array:
    n_eval, d = X_eval.shape
    n_chunks = (n_eval + chunk_size - 1) // chunk_size
    padded_n = n_chunks * chunk_size
    pad_rows = padded_n - n_eval

    X_eval_padded = jnp.pad(X_eval, ((0, pad_rows), (0, 0)))
    valid_mask = jnp.pad(jnp.ones((n_eval,), dtype=X_eval.dtype), (0, pad_rows))

    X_eval_chunks = X_eval_padded.reshape(n_chunks, chunk_size, d)
    mask_chunks = valid_mask.reshape(n_chunks, chunk_size)

    def chunk_contribution(X_chunk: Array, mask_chunk: Array) -> tuple[Array, Array]:
        sensitivities = local_sensitivity_batch(
            U_star, X_chunk, X_proj, f_vals, rank, include_beta_jacobian, ridge
        )
        quadratic_terms = jnp.einsum("ni,ij,nj->n", sensitivities, sigma_z, sensitivities)
        return jnp.sum(quadratic_terms * mask_chunk), jnp.sum(mask_chunk)

    chunk_sums, chunk_counts = jax.vmap(chunk_contribution)(X_eval_chunks, mask_chunks)
    return jnp.sum(chunk_sums) / jnp.sum(chunk_counts)


@partial(
    jax.jit,
    static_argnames=("rank", "n_proj", "n_eval", "proj_chunk_size", "eval_chunk_size", "include_beta_jacobian"),
)
def quadratic_risk_from_sigma_fully_streaming(
    U_star: Array,
    sigma_z: Array,
    theta_star: Array,
    key_proj: Array,
    key_eval: Array,
    n_proj: int,
    n_eval: int,
    rank: int,
    proj_chunk_size: int,
    eval_chunk_size: int,
    include_beta_jacobian: bool = True,
    ridge: float = 1e-6,
) -> Array:
    n_chunks = (n_eval + eval_chunk_size - 1) // eval_chunk_size

    def body_fun(idx: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        total_sum, total_count = carry
        chunk_key = jr.fold_in(key_eval, idx)
        X_chunk, _ = sample_mixture(chunk_key, U_star, eval_chunk_size)
        remaining = n_eval - idx * eval_chunk_size
        valid_count = jnp.minimum(eval_chunk_size, remaining)
        mask = (jnp.arange(eval_chunk_size) < valid_count).astype(U_star.dtype)

        sensitivities = local_sensitivity_batch_streaming_proj(
            U_star,
            X_chunk,
            theta_star,
            key_proj,
            n_proj,
            rank,
            proj_chunk_size,
            include_beta_jacobian,
            ridge,
        )
        quadratic_terms = jnp.einsum("ni,ij,nj->n", sensitivities, sigma_z, sensitivities)
        return total_sum + jnp.sum(quadratic_terms * mask), total_count + valid_count

    total_sum, total_count = jax.lax.fori_loop(
        0,
        n_chunks,
        body_fun,
        (jnp.array(0.0, dtype=U_star.dtype), jnp.array(0, dtype=jnp.int32)),
    )
    return total_sum / total_count.astype(U_star.dtype)


@partial(jax.jit, static_argnames=("rank", "n_eval", "chunk_size", "include_beta_jacobian"))
def quadratic_risk_from_sigma_streaming(
    U_star: Array,
    sigma_z: Array,
    X_proj: Array,
    f_vals: Array,
    key_eval: Array,
    n_eval: int,
    rank: int,
    chunk_size: int,
    include_beta_jacobian: bool = True,
    ridge: float = 1e-6,
) -> Array:
    n_chunks = (n_eval + chunk_size - 1) // chunk_size

    def body_fun(idx: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        total_sum, total_count = carry
        chunk_key = jr.fold_in(key_eval, idx)
        X_chunk, _ = sample_mixture(chunk_key, U_star, chunk_size)
        remaining = n_eval - idx * chunk_size
        valid_count = jnp.minimum(chunk_size, remaining)
        mask = (jnp.arange(chunk_size) < valid_count).astype(U_star.dtype)

        sensitivities = local_sensitivity_batch(
            U_star, X_chunk, X_proj, f_vals, rank, include_beta_jacobian, ridge
        )
        quadratic_terms = jnp.einsum("ni,ij,nj->n", sensitivities, sigma_z, sensitivities)
        return total_sum + jnp.sum(quadratic_terms * mask), total_count + valid_count

    total_sum, total_count = jax.lax.fori_loop(
        0,
        n_chunks,
        body_fun,
        (jnp.array(0.0, dtype=U_star.dtype), jnp.array(0, dtype=jnp.int32)),
    )
    return total_sum / total_count.astype(U_star.dtype)


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
    eval_chunk_size: int | None = 256,
    proj_chunk_size: int | None = None,
    include_beta_jacobian: bool = True,
    store_samples_in_diagnostics: bool = False,
) -> tuple[Array, dict[str, Array]]:
    U_star = jnp.asarray(U_star, dtype=jnp.float32)
    theta_star = jnp.asarray(theta_star, dtype=jnp.float32)
    rank = infer_subspace_rank(U_star) if rank is None else rank

    key_fisher, key_proj, key_eval = jr.split(key, 3)

    fisher_hat = estimate_fisher_information(U_star, key_fisher, n_fisher_mc)
    sigma_z = regularized_inverse(fisher_hat, fisher_ridge)

    X_proj = None
    X_eval = None
    f_vals = None

    if proj_chunk_size is None:
        X_proj, _ = sample_mixture(key_proj, U_star, n_proj_mc)
        f_vals = predict_batch(X_proj, U_star, theta_star, rank)

    if proj_chunk_size is None and eval_chunk_size is None:
        X_eval, _ = sample_mixture(key_eval, U_star, n_eval_mc)
        risk = quadratic_risk_from_sigma(
            U_star, sigma_z, X_proj, f_vals, X_eval, rank, include_beta_jacobian, proj_ridge
        )
    elif proj_chunk_size is None:
        risk = quadratic_risk_from_sigma_streaming(
            U_star,
            sigma_z,
            X_proj,
            f_vals,
            key_eval,
            n_eval_mc,
            rank,
            eval_chunk_size,
            include_beta_jacobian,
            proj_ridge,
        )
    else:
        if eval_chunk_size is None:
            raise ValueError("eval_chunk_size must be set when proj_chunk_size is used.")
        risk = quadratic_risk_from_sigma_fully_streaming(
            U_star,
            sigma_z,
            theta_star,
            key_proj,
            key_eval,
            n_proj_mc,
            n_eval_mc,
            rank,
            proj_chunk_size,
            eval_chunk_size,
            include_beta_jacobian,
            proj_ridge,
        )

    diagnostics = {
        "fisher_hat": fisher_hat,
        "sigma_z": sigma_z,
    }
    if f_vals is not None:
        diagnostics["f_vals"] = f_vals
    if store_samples_in_diagnostics:
        if X_proj is not None:
            diagnostics["X_proj"] = X_proj
        if X_eval is not None:
            diagnostics["X_eval"] = X_eval
    return risk, diagnostics
