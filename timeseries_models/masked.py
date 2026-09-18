"""State-space models with missing observations.

Missing entries of the observation matrix are handled exactly by an
information-form Kalman update: unobserved dimensions contribute zero
precision, so a time step with no observed dimension propagates the state
model prediction unchanged (no observation update). The observation noise is
constrained to be diagonal, which keeps every quantity fixed-shape (JAX
scan/vmap friendly) and well-conditioned in float32.

Usage:
    om = MaskedLinearObservationModel(Dx, Dz)
    sm = LSEMStateModel(Dz, Dk)
    ssm = MaskedStateSpaceModel(om, sm)
    ssm.fit(X, mask)              # X, mask: [B, T, Dx]; mask 1=observed
    ssm.compute_predictive_log_likelihood(X_test, mask_test)
    smooth = ssm.smooth(X, mask)  # smoothed latent density dict
"""
from typing import Tuple
import time

import jax
from jax import lax, jit, vmap, pmap
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from tqdm import tqdm

from gaussian_toolbox import pdf
from timeseries_models import observation_model, state_model, state_space_model

QX_FLOOR = 1e-6
_DICT_AXES = {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0}


def _shard_batch(arrs, n_dev):
    """Split leading batch axis across devices; returns sharded arrays and
    per-window weights (0 for padding copies)."""
    B = jax.tree_util.tree_leaves(arrs[0])[0].shape[0]
    pad = (-B) % n_dev
    w = jnp.concatenate([jnp.ones(B), jnp.zeros(pad)]).reshape(n_dev, -1)

    def shard(a):
        if pad:
            a = jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)])
        return a.reshape((n_dev, (B + pad) // n_dev) + a.shape[1:])
    return [jax.tree_util.tree_map(shard, a) for a in arrs], w, B


class ShardedLSEMStateModel(state_model.LSEMStateModel):
    """LSEM state model whose M-step sufficient statistics and kernel-weight
    objective are computed data-parallel over local devices (windows are
    independent given the parameters).

    `kernel_maxiter` bounds the L-BFGS-B iterations of the kernel-weight update
    per EM step (a partial M-step still yields a generalized EM)."""
    kernel_maxiter = 100

    def _sharded_sum(self, stats_fn, *dicts):
        n_dev = jax.local_device_count()
        B = dicts[0]["mu"].shape[0]
        if n_dev == 1 or B < n_dev:
            return [jnp.sum(a, axis=0) for a in jit(vmap(stats_fn))(*dicts)]
        sh, w, _ = _shard_batch(dicts, n_dev)
        axes = [_DICT_AXES] * len(dicts) + [0]

        def per_device(*args):
            *ds, w_d = args
            out = vmap(lambda *a: [x * a[-1] for x in stats_fn(*a[:-1])],
                       in_axes=axes)(*ds, w_d)
            return [jnp.sum(x, axis=0) for x in out]
        out = pmap(per_device)(*sh, w)
        return [jnp.sum(x, axis=0) for x in out]

    def _update_A(self, smooth_dict, two_step_smooth_dict):
        T, Eff, Ezf, Ef = self._sharded_sum(self._get_A_stats, smooth_dict,
                                            two_step_smooth_dict)
        Ebf = (Ef[None] * self.b[:, None]) / T
        return jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T

    def _update_b(self, smooth_dict):
        T, Ef, sd_mu = self._sharded_sum(self._get_b_stats, smooth_dict)
        return sd_mu / T - jnp.dot(self.A, Ef / T).T

    def _update_Qz(self, smooth_dict, two_step_smooth_dict, **kwargs):
        T, Qz = self._sharded_sum(self._get_Qz_stats, smooth_dict,
                                  two_step_smooth_dict)
        return Qz / T

    def update_hyperparameters(self, smooth_dict, two_step_smooth_dict, **kwargs):
        self.A = self._update_A(smooth_dict, two_step_smooth_dict)
        self.Qz = self._update_Qz(smooth_dict, two_step_smooth_dict)
        self.Lz = self.mat_to_cholvec(self.Qz)
        self.b = self._update_b(smooth_dict)
        self.update_state_density()
        self._update_kernel_params(smooth_dict, two_step_smooth_dict)
        self.update_state_density()

    def _update_kernel_params(self, smooth_dict, two_step_smooth_dict):
        n_dev = jax.local_device_count()
        B = smooth_dict["mu"].shape[0]
        if n_dev == 1 or B < n_dev:
            return super()._update_kernel_params(smooth_dict, two_step_smooth_dict)
        (sd, tsd), w, B = _shard_batch([smooth_dict, two_step_smooth_dict], n_dev)

        def objective(W, sd_i, tsd_i, w_i):
            self.state_density.w0 = W[:, 0]
            self.state_density.W = W[:, 1:]
            self.state_density.update_phi()
            return -w_i * self.compute_Q_function(pdf.GaussianPDF(**sd_i),
                                                 pdf.GaussianPDF(**tsd_i))

        def per_device(W, sd_d, tsd_d, w_d):
            return jnp.sum(vmap(objective, in_axes=[None, _DICT_AXES,
                                                    _DICT_AXES, 0])(W, sd_d, tsd_d, w_d))
        # pmap must stay outermost (pmap inside jit does not lower), so value
        # and gradient are computed per device and reduced on the host before
        # being handed to scipy's L-BFGS-B.
        p_vg = pmap(jax.value_and_grad(per_device))
        shape, lam, dtype = self.W.shape, self.lambda_W, self.W.dtype

        def fun_jac(W_flat):
            W = jnp.asarray(W_flat, dtype=dtype).reshape(shape)
            W_b = jnp.broadcast_to(W, (n_dev,) + shape)
            vals, grads = p_vg(W_b, sd, tsd, w)
            f = float(jnp.sum(vals)) / B + 0.5 * lam * float(jnp.sum(W ** 2))
            g = np.asarray(jnp.sum(grads, axis=0)) / B + lam * np.asarray(W)
            return f, g.ravel().astype(np.float64)
        res = scipy_minimize(fun_jac, np.asarray(self.W, dtype=np.float64).ravel(),
                             jac=True, method="L-BFGS-B",
                             options={"maxiter": self.kernel_maxiter})
        self.W = jnp.asarray(res.x.reshape(shape), dtype=dtype)


def _logdet_psd(M):
    L = jnp.linalg.cholesky(M)
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)


class MaskedLinearObservationModel(observation_model.LinearObservationModel):
    """Linear-Gaussian observation model x_t = C z_t + d + xi_t with diagonal
    noise and per-entry missingness masks.

    `qx_floor` (scalar or per-channel array) lower-bounds the noise variances
    in the M-step, e.g. at the measurement-error level, which prevents the
    degenerate zero-noise (Heywood) solution."""
    qx_floor = QX_FLOOR

    def __init__(self, Dx: int, Dz: int, noise_x: float = 1.0, delta: float = 0,
                 key=None):
        kwargs = {} if key is None else {"key": key}
        super().__init__(Dx, Dz, noise_x=noise_x, delta=delta, **kwargs)

    @property
    def qx(self):
        return jnp.diagonal(self.Qx) + self.delta

    def filtering(self, prediction_density: pdf.GaussianPDF, x_t: jnp.ndarray,
                  mask_t: jnp.ndarray = None, **kwargs) -> pdf.GaussianPDF:
        if mask_t is None:
            return super().filtering(prediction_density, x_t, **kwargs)
        mu_p, Lambda_p = prediction_density.mu[0], prediction_density.Lambda[0]
        m = mask_t[0]
        x = jnp.nan_to_num(x_t[0]) * m
        w = m / self.qx
        CtW = self.C.T * w[None]                       # (Dz, Dx)
        Lambda_f = Lambda_p + CtW @ self.C
        Lambda_f = 0.5 * (Lambda_f + Lambda_f.T)
        Sigma_f = jnp.linalg.inv(Lambda_f)
        Sigma_f = 0.5 * (Sigma_f + Sigma_f.T)
        mu_f = Sigma_f @ (Lambda_p @ mu_p + CtW @ (x - self.d * m))
        return pdf.GaussianPDF(Sigma=Sigma_f[None], mu=mu_f[None],
                               Lambda=Lambda_f[None],
                               ln_det_Sigma=-_logdet_psd(Lambda_f)[None])

    def masked_predictive_llk(self, prediction_density: pdf.GaussianPDF,
                              x_t: jnp.ndarray, mask_t: jnp.ndarray):
        """log N(x_obs; C mu_p + d, C Sigma_p C' + Q) over observed dims only,
        via the matrix determinant lemma / Woodbury (fixed shapes)."""
        mu_p, Lambda_p = prediction_density.mu[0], prediction_density.Lambda[0]
        m = mask_t[0]
        q = self.qx
        w = m / q
        r = (jnp.nan_to_num(x_t[0]) - self.C @ mu_p - self.d) * m
        Lambda_f = Lambda_p + (self.C.T * w[None]) @ self.C
        Lambda_f = 0.5 * (Lambda_f + Lambda_f.T)
        Sigma_f = jnp.linalg.inv(Lambda_f)
        v = self.C.T @ (w * r)
        quad = jnp.sum(w * r * r) - v @ Sigma_f @ v
        logdet = (jnp.sum(m * jnp.log(q)) + _logdet_psd(Lambda_f)
                  - _logdet_psd(Lambda_p))
        n_obs = jnp.sum(m)
        return -0.5 * (n_obs * jnp.log(2 * jnp.pi) + logdet + quad)

    def update_hyperparameters(self, X: jnp.ndarray, smooth_dict: dict,
                               mask: jnp.ndarray = None, **kwargs):
        if mask is None:
            return super().update_hyperparameters(X, smooth_dict, **kwargs)
        C, d, qx = jit(self._masked_mstep)(X, mask, smooth_dict)
        self.C, self.d = C, d
        self.Qx = jnp.diag(qx)
        self.Lx = self.mat_to_cholvec(self.Qx)
        self.update_observation_density()

    def _masked_mstep(self, X, mask, smooth_dict):
        mu = smooth_dict["mu"]              # (B, T, Dz)
        Sigma = smooth_dict["Sigma"]        # (B, T, Dz, Dz)
        Xm = jnp.nan_to_num(X) * mask
        Ezz = Sigma + mu[..., :, None] * mu[..., None, :]
        ones = jnp.ones(mu.shape[:-1] + (1,))
        phi = jnp.concatenate([mu, ones], axis=-1)                 # (B,T,Dz+1)
        Ephi = jnp.concatenate([
            jnp.concatenate([Ezz, mu[..., :, None]], axis=-1),
            jnp.concatenate([mu[..., None, :], ones[..., None]], axis=-1)],
            axis=-2)                                               # (B,T,Dz+1,Dz+1)
        A = jnp.einsum("bti,btjk->ijk", mask, Ephi)                # (Dx,Dz+1,Dz+1)
        b = jnp.einsum("bti,btj->ij", Xm, phi)                     # (Dx,Dz+1)
        A = A + 1e-6 * jnp.eye(A.shape[-1])[None]
        wts = jnp.linalg.solve(A, b[..., None])[..., 0]            # (Dx,Dz+1)
        C, d = wts[:, :-1], wts[:, -1]
        pred = jnp.einsum("btj,ij->bti", mu, C) + d
        var = jnp.einsum("ij,btjk,ik->bti", C, Sigma, C)
        resid2 = ((Xm - pred * mask) ** 2 + var) * mask
        n = jnp.sum(mask, axis=(0, 1))
        qx = jnp.sum(resid2, axis=(0, 1)) / jnp.maximum(n, 1.0)
        qx = jnp.maximum(qx, self.qx_floor)
        return C, d, qx


def _sharded_map(fn, arrs, n_out):
    """Apply a per-window function over the batch, data-parallel over local
    devices; returns a list of n_out arrays with leading batch axis."""
    n_dev = jax.local_device_count()
    B = jax.tree_util.tree_leaves(arrs[0])[0].shape[0]
    if n_dev == 1 or B < n_dev:
        out = jit(vmap(fn))(*arrs)
        return list(out) if n_out > 1 else [out]
    sh, _, _ = _shard_batch(arrs, n_dev)
    out = pmap(vmap(fn))(*sh)
    out = list(out) if n_out > 1 else [out]
    return [o.reshape((-1,) + o.shape[2:])[:B] for o in out]


class MaskedLSEMObservationModel(observation_model.LSEMObservationModel):
    """Projected-kernel (LSEM) observation model x_t = C phi(z_t) + d + xi_t
    with diagonal noise and per-entry missingness masks.

    Filtering conditions the moment-matched joint Gaussian p(z_t, x_t) on the
    observed dimensions only, exactly and with fixed shapes: masked rows and
    columns of Sigma_xx are replaced by an identity block with zero residual
    and zero cross-covariance, which decouples them from the update.
    `qx_floor` lower-bounds the noise variances (see the linear model)."""
    kernel_maxiter = 100
    qx_floor = QX_FLOOR

    def __init__(self, Dx: int, Dz: int, Dk: int, noise_x: float = 1.0,
                 lambda_W: float = 0.0, key=None):
        kwargs = {} if key is None else {"key": key}
        super().__init__(Dx, Dz, Dk, noise_x=noise_x, lambda_W=lambda_W, **kwargs)

    @property
    def qx(self):
        return jnp.diagonal(self.Qx)

    # ---- inference --------------------------------------------------------
    def _masked_joint_terms(self, prediction_density, x_t, mask_t):
        joint = self.observation_density.affine_joint_transformation(
            prediction_density)
        Dz = self.Dz
        mu, S = joint.mu[0], joint.Sigma[0]
        m = mask_t[0]
        mu_z, mu_x = mu[:Dz], mu[Dz:]
        Szz, Szx, Sxx = S[:Dz, :Dz], S[:Dz, Dz:], S[Dz:, Dz:]
        Sxx_m = Sxx * (m[:, None] * m[None]) + jnp.diag(1.0 - m)
        Sxx_m = 0.5 * (Sxx_m + Sxx_m.T)
        Szx_m = Szx * m[None]
        r = m * (jnp.nan_to_num(x_t[0]) - mu_x)
        return mu_z, Szz, Szx_m, Sxx_m, r, m

    def filtering(self, prediction_density: pdf.GaussianPDF, x_t: jnp.ndarray,
                  mask_t: jnp.ndarray = None, **kwargs) -> pdf.GaussianPDF:
        if mask_t is None:
            return super().filtering(prediction_density, x_t, **kwargs)
        mu_z, Szz, Szx_m, Sxx_m, r, _ = self._masked_joint_terms(
            prediction_density, x_t, mask_t)
        K = jnp.linalg.solve(Sxx_m, Szx_m.T).T
        mu_f = mu_z + K @ r
        Sigma_f = Szz - K @ Szx_m.T
        Sigma_f = 0.5 * (Sigma_f + Sigma_f.T)
        return pdf.GaussianPDF(Sigma=Sigma_f[None], mu=mu_f[None])

    def masked_predictive_llk(self, prediction_density: pdf.GaussianPDF,
                              x_t: jnp.ndarray, mask_t: jnp.ndarray):
        _, _, _, Sxx_m, r, m = self._masked_joint_terms(
            prediction_density, x_t, mask_t)
        quad = r @ jnp.linalg.solve(Sxx_m, r)
        return -0.5 * (jnp.sum(m) * jnp.log(2 * jnp.pi) + _logdet_psd(Sxx_m) + quad)

    # ---- M-step -----------------------------------------------------------
    def _phi_moments(self, W, smooth_dict):
        """E[phi~], E[phi~ phi~'] per time step for one window under the
        smoothing density, with phi~ = (z, k(h_1(z)), ..., k(h_Dk(z)), 1)."""
        from gaussian_toolbox import factor
        p = pdf.GaussianPDF(**smooth_dict)
        w0, Wz = W[:, 0], W[:, 1:]
        k_func = factor.OneRankFactor(v=Wz, nu=Wz * w0[:, None],
                                      ln_beta=-0.5 * w0 ** 2)
        R, Dk, Dz = p.R, self.Dk, self.Dz
        Ex = p.integrate("x")
        p_k = p.multiply(k_func, update_full=True)
        Ek = p_k.integrate().reshape((R, Dk))
        Exx = p.integrate("xx'")
        Ekx = p_k.integrate("x").reshape((R, Dk, Dz))
        Ekk = p_k.multiply(k_func, update_full=True).integrate().reshape((R, Dk, Dk))
        Ef = jnp.concatenate([Ex, Ek, jnp.ones((R, 1))], axis=1)
        Eff = jnp.block([[Exx, jnp.swapaxes(Ekx, 1, 2)], [Ekx, Ekk]])
        Ef0 = Ef[:, :-1]
        Eff = jnp.concatenate([
            jnp.concatenate([Eff, Ef0[:, :, None]], axis=2),
            jnp.concatenate([Ef0[:, None, :], jnp.ones((R, 1, 1))], axis=2)], axis=1)
        return Ef, Eff

    def _masked_quadratic(self, Ef, Eff, Xm, mask, Cd):
        """sum_t m_ti E[(x_ti - cd_i' phi~_t)^2] for every observation dim i."""
        lin = jnp.einsum("bti,btj,ij->i", Xm, Ef, Cd)
        quad = jnp.einsum("bti,ij,btjk,ik->i", mask, Cd, Eff, Cd)
        return jnp.sum(Xm ** 2, axis=(0, 1)) - 2.0 * lin + quad

    def update_hyperparameters(self, X: jnp.ndarray, smooth_dict: dict,
                               mask: jnp.ndarray = None, **kwargs):
        if mask is None:
            return super().update_hyperparameters(X, smooth_dict, **kwargs)
        Xm = jnp.nan_to_num(X) * mask
        W_full = jnp.concatenate([self.observation_density.w0[:, None],
                                  self.observation_density.W], axis=1)
        Ef, Eff = _sharded_map(lambda sd: self._phi_moments(W_full, sd),
                               [smooth_dict], 2)
        # closed-form (C, d) per observation dimension, then diagonal noise
        A = jnp.einsum("bti,btjk->ijk", mask, Eff) \
            + 1e-6 * jnp.eye(Eff.shape[-1])[None]
        b = jnp.einsum("bti,btj->ij", Xm, Ef)
        Cd = jnp.linalg.solve(A, b[..., None])[..., 0]
        n = jnp.maximum(jnp.sum(mask, axis=(0, 1)), 1.0)
        qx = jnp.maximum(self._masked_quadratic(Ef, Eff, Xm, mask, Cd) / n,
                         self.qx_floor)
        self.C, self.d = Cd[:, :-1], Cd[:, -1]
        self.Qx = jnp.diag(qx)
        self.Lx = jnp.linalg.cholesky(self.Qx)
        self.update_observation_density()
        self._update_kernel_params_masked(Xm, mask, smooth_dict, Cd, qx)
        self.update_observation_density()

    def _update_kernel_params_masked(self, Xm, mask, smooth_dict, Cd, qx):
        n_dev = jax.local_device_count()
        B = Xm.shape[0]
        n_dev = n_dev if B >= n_dev else 1
        (sd, Xs, Ms), w, _ = _shard_batch([smooth_dict, Xm, mask], n_dev)

        def per_window(W, sd_i, X_i, M_i, w_i):
            Ef, Eff = self._phi_moments(W, sd_i)
            q = self._masked_quadratic(Ef[None], Eff[None], X_i[None], M_i[None], Cd)
            return w_i * 0.5 * jnp.sum(q / qx)

        def per_device(W, sd_d, X_d, M_d, w_d):
            return jnp.sum(vmap(per_window, in_axes=[None, _DICT_AXES, 0, 0, 0])(
                W, sd_d, X_d, M_d, w_d))
        p_vg = pmap(jax.value_and_grad(per_device))
        W0 = jnp.concatenate([self.observation_density.w0[:, None],
                              self.observation_density.W], axis=1)
        shape, lam, dtype = W0.shape, self.lambda_W, W0.dtype

        def fun_jac(W_flat):
            W = jnp.asarray(W_flat, dtype=dtype).reshape(shape)
            W_b = jnp.broadcast_to(W, (n_dev,) + shape)
            vals, grads = p_vg(W_b, sd, Xs, Ms, w)
            f = float(jnp.sum(vals)) / B + 0.5 * lam * float(jnp.sum(W ** 2))
            g = np.asarray(jnp.sum(grads, axis=0)) / B + lam * np.asarray(W)
            return f, g.ravel().astype(np.float64)
        res = scipy_minimize(fun_jac, np.asarray(W0, dtype=np.float64).ravel(),
                             jac=True, method="L-BFGS-B",
                             options={"maxiter": self.kernel_maxiter})
        self.W = jnp.asarray(res.x.reshape(shape), dtype=dtype)


class MaskedStateSpaceModel(state_space_model.StateSpaceModel):
    """StateSpaceModel whose E-step, M-step and likelihood accept a
    missingness mask. Control inputs are not supported."""

    def _init_masked(self, X, mask, mu0=None, Sigma0=None):
        X = jnp.asarray(X)
        if X.ndim == 2:
            X = X[None]
        if mask is None:
            mask = jnp.isfinite(X).astype(X.dtype)
        mask = jnp.asarray(mask, dtype=X.dtype)
        if mask.ndim == 2:
            mask = mask[None]
        assert mask.shape == X.shape
        B, T = X.shape[:2]
        if mu0 is None:
            mu0 = jnp.zeros((B, 1, self.sm.Dz))
        if Sigma0 is None:
            Sigma0 = jnp.tile(jnp.eye(self.sm.Dz)[None, None], (B, 1, 1, 1))
        control_z = jnp.empty((B, T, 0))
        return X, mask, mu0, Sigma0, control_z

    # ---- E-step -----------------------------------------------------------
    def _forward_step_masked(self, carry, vars_t):
        X_t, mask_t = vars_t
        pred = self.sm.prediction(carry)
        filt = self.om.filtering(pred, X_t[None], mask_t=mask_t[None])
        return filt, (filt.Sigma[0], filt.mu[0], filt.Lambda[0],
                      filt.ln_det_Sigma[0])

    def _forward_sweep_masked(self, X, mask, mu0, Sigma0):
        pz0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
        init = self.om.filtering(pz0, X[:1], mask_t=mask[:1])
        _, res = lax.scan(self._forward_step_masked, init, (X[1:], mask[1:]))
        Sigma_f, mu_f, Lambda_f, ln_det_f = res
        return {
            "Sigma": jnp.concatenate([init.Sigma, Sigma_f]),
            "mu": jnp.concatenate([init.mu, mu_f]),
            "Lambda": jnp.concatenate([init.Lambda, Lambda_f]),
            "ln_det_Sigma": jnp.concatenate([init.ln_det_Sigma, ln_det_f]),
        }

    def _estep_masked(self, X, mask, mu0, Sigma0, control_z):
        filter_dict = self._forward_sweep_masked(X, mask, mu0, Sigma0)
        return self._backward_sweep(X, filter_dict, control_z)

    def estep_masked(self, X, mask, mu0, Sigma0, control_z):
        """E-step over the batch; data-parallel over all local JAX devices
        (windows are independent given the parameters). On CPU, expose one
        device per core with XLA_FLAGS=--xla_force_host_platform_device_count=N."""
        n_dev = jax.local_device_count()
        B = X.shape[0]
        if n_dev == 1 or B < n_dev:
            return jit(vmap(self._estep_masked))(X, mask, mu0, Sigma0, control_z)
        pad = (-B) % n_dev
        def shard(a):
            if pad:
                a = jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)])
            return a.reshape((n_dev, (B + pad) // n_dev) + a.shape[1:])
        args = [shard(a) for a in (X, mask, mu0, Sigma0, control_z)]
        smooth, two_step = pmap(vmap(self._estep_masked))(*args)
        unshard = lambda d: {k: v.reshape((-1,) + v.shape[2:])[:B]
                             for k, v in d.items()}
        return unshard(smooth), unshard(two_step)

    _PARAM_NAMES = ("C", "d", "Qx", "Lx", "W", "A", "b", "Qz", "Lz")

    def _snapshot_params(self):
        return [{k: getattr(m, k) for k in self._PARAM_NAMES if hasattr(m, k)}
                for m in (self.om, self.sm)]

    def _restore_params(self, snapshot):
        for m, snap in zip((self.om, self.sm), snapshot):
            for k, v in snap.items():
                setattr(m, k, v)
        self.om.update_observation_density()
        self.sm.update_state_density()

    # ---- fit --------------------------------------------------------------
    def fit(self, X, mask=None, max_iter: int = 100, conv_crit: float = 1e-3,
            progress: bool = True, **kwargs):
        X, mask, mu0, Sigma0, control_z = self._init_masked(X, mask)
        llk_list = []
        converged, it = False, 0
        self.timing = {"estep": 0.0, "mstep_sm": 0.0, "mstep_om": 0.0, "llk": 0.0}
        pbar = tqdm(total=max_iter, desc="EM (masked)", disable=not progress,
                    dynamic_ncols=True)
        self.fit_status = {"nan_at_iter": None}
        while it < max_iter and not converged:
            snapshot = self._snapshot_params()
            prev = (mu0, Sigma0)
            t0 = time.perf_counter()
            smooth, two_step = self.estep_masked(X, mask, mu0, Sigma0, control_z)
            smooth["mu"].block_until_ready()
            self.timing["estep"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            self.sm.update_hyperparameters(smooth, two_step, control_z=control_z)
            self.timing["mstep_sm"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            self.om.update_hyperparameters(X, smooth, mask=mask)
            self.timing["mstep_om"] += time.perf_counter() - t0
            mu0, Sigma0 = smooth["mu"][:, :1], smooth["Sigma"][:, :1]
            t0 = time.perf_counter()
            llk = float(self.compute_predictive_log_likelihood(
                X, mask, mu0, Sigma0))
            self.timing["llk"] += time.perf_counter() - t0
            if not np.isfinite(llk):
                # numerical breakdown: roll back to the last finite parameters
                self._restore_params(snapshot)
                mu0, Sigma0 = prev
                self.fit_status["nan_at_iter"] = it
                smooth, two_step = self.estep_masked(X, mask, mu0, Sigma0,
                                                     control_z)
                break
            llk_list.append(llk)
            if it > 2:
                converged = bool(self._check_convergence(
                    llk_list[-2], llk, conv_crit))
            pbar.set_postfix(iteration=it, log_likelihood=llk)
            pbar.update(1)
            it += 1
        pbar.close()
        self.p0 = {"mu": mu0, "Sigma": Sigma0}
        return llk_list, self.p0, smooth, two_step

    # ---- likelihood / smoothing ------------------------------------------
    def _llk_step(self, carry, vars_t):
        X_t, mask_t = vars_t
        filt = carry
        pred = self.sm.prediction(filt)
        llk_t = self.om.masked_predictive_llk(pred, X_t[None], mask_t[None])
        filt = self.om.filtering(pred, X_t[None], mask_t=mask_t[None])
        return filt, llk_t

    def _predictive_llk_single(self, X, mask, mu0, Sigma0):
        pz0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
        llk0 = self.om.masked_predictive_llk(pz0, X[:1], mask[:1])
        init = self.om.filtering(pz0, X[:1], mask_t=mask[:1])
        _, llk_t = lax.scan(self._llk_step, init, (X[1:], mask[1:]))
        return llk0 + jnp.sum(llk_t)

    def compute_predictive_log_likelihood(self, X, mask=None, mu0=None,
                                          Sigma0=None, **kwargs):
        X, mask, mu0, Sigma0, _ = self._init_masked(X, mask, mu0, Sigma0)
        n_dev, B = jax.local_device_count(), X.shape[0]
        if n_dev == 1 or B < n_dev:
            return jnp.sum(jit(vmap(self._predictive_llk_single))(
                X, mask, mu0, Sigma0))
        pad = (-B) % n_dev
        def shard(a):
            if pad:
                a = jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)])
            return a.reshape((n_dev, (B + pad) // n_dev) + a.shape[1:])
        llk = pmap(vmap(self._predictive_llk_single))(
            *[shard(a) for a in (X, mask, mu0, Sigma0)])
        return jnp.sum(llk.reshape(-1)[:B])

    def smooth(self, X, mask=None, mu0=None, Sigma0=None):
        X, mask, mu0, Sigma0, control_z = self._init_masked(X, mask, mu0, Sigma0)
        smooth, _ = self.estep_masked(X, mask, mu0, Sigma0, control_z)
        return {k: np.asarray(v) for k, v in smooth.items()}
