"""Regression for mixed-precision expected residuals in masked LSEM M-steps.

The posterior is frozen, so a covariance update must increase its true Q
function even though the model's approximate predictive likelihood need not.
"""
import unittest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from timeseries_models.masked import MaskedLSEMObservationModel


class _FrozenKernelObservation(MaskedLSEMObservationModel):
    def _update_kernel_params_masked(self, *args, **kwargs):
        # Isolate the closed-form (C, d, Qx) coordinate step.
        pass


class TestMaskedLSEMPrecision(unittest.TestCase):
    @staticmethod
    def _model():
        model = _FrozenKernelObservation(1, 1, 2)
        model.W = jnp.asarray([[0.3, 0.5], [-0.8, -1.1]], dtype=jnp.float64)
        model.update_observation_density()
        return model

    @staticmethod
    def _residual(model, moments, x, mask):
        # Evaluate the expectation without subtracting large accumulated sums.
        ef, eff = moments
        cd = jnp.concatenate([model.C, model.d[:, None]], axis=1)
        mean = jnp.einsum('btj,ij->bti', ef, cd)
        covariance = eff - ef[..., :, None] * ef[..., None, :]
        variance = jnp.einsum('ij,btjk,ik->bti', cd, covariance, cd)
        return jnp.sum(mask * ((jnp.nan_to_num(x) - mean) ** 2 + variance), axis=(0, 1))

    def test_float32_observations_preserve_closed_form_noise_update_and_Q(self):
        for offset in (0.0, 10.0):
            with self.subTest(offset=offset):
                rng = np.random.default_rng(0)
                x = (offset + rng.normal(size=(2, 512, 1))).astype(np.float32)
                mask = np.ones_like(x)
                mask.reshape(-1)[::7] = 0
                sd = {'mu': jnp.asarray(x, dtype=jnp.float64),
                      'Sigma': jnp.full((2, 512, 1, 1), 1e-5, dtype=jnp.float64)}
                x32 = jnp.asarray(np.where(mask > 0, x, np.nan))
                m32 = jnp.asarray(mask)
                x64, m64 = x32.astype(jnp.float64), m32.astype(jnp.float64)
                n = jnp.sum(m64, axis=(0, 1))

                reference = self._model()
                reference.update_hyperparameters(x64, sd, mask=m64)
                moments = jax.vmap(lambda s: reference._phi_moments(reference.W, s))(sd)
                r_before = self._residual(reference, moments, x64, m64)
                q_optimum = jnp.maximum(r_before / n, reference.qx_floor)
                reference.Qx = jnp.diag(q_optimum)
                reference.update_observation_density()
                Q_before = float(-0.5 * jnp.sum(n * jnp.log(2 * jnp.pi * q_optimum)
                                                 + r_before / q_optimum))

                mixed = self._model()
                mixed.C, mixed.d, mixed.Qx = reference.C, reference.d, reference.Qx
                mixed.update_observation_density()
                mixed.update_hyperparameters(x32, sd, mask=m32)
                np.testing.assert_array_equal(np.asarray(mixed.C), np.asarray(reference.C))
                np.testing.assert_array_equal(np.asarray(mixed.d), np.asarray(reference.d))
                q_after = jnp.diagonal(mixed.Qx)
                np.testing.assert_allclose(np.asarray(q_after), np.asarray(q_optimum),
                                           rtol=1e-6, atol=1e-11)
                r_after = self._residual(mixed, moments, x64, m64)
                Q_after = float(-0.5 * jnp.sum(n * jnp.log(2 * jnp.pi * q_after)
                                                + r_after / q_after))
                self.assertGreaterEqual(Q_after, Q_before - 1e-6)

    def test_all_float32_quadratic_keeps_float32_output(self):
        model = self._model()
        ef = jnp.asarray([[[0.3, 1.0]]], dtype=jnp.float32)
        eff = jnp.asarray([[[[0.5, 0.3], [0.3, 1.0]]]], dtype=jnp.float32)
        x = jnp.asarray([[[0.6]]], dtype=jnp.float32)
        mask = jnp.ones_like(x)
        cd = jnp.asarray([[0.7, 0.2]], dtype=jnp.float32)
        result = model._masked_quadratic(ef, eff, x, mask, cd)
        self.assertEqual(result.dtype, jnp.dtype('float32'))
        # E[(0.6 - 0.7*z - 0.2)^2], E[z]=0.3, E[z^2]=0.5.
        np.testing.assert_allclose(np.asarray(result), [0.237], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
