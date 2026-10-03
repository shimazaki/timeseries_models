"""A deterministic counterexample to exact-EM monotonicity for LSEM inference.

This uses only upstream classes. Run through Slurm on shared clusters.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gaussian_toolbox import pdf
from timeseries_models.observation_model import LSEMObservationModel


class TestApproximateEM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", cls._previous_x64)

    def test_exact_variance_mstep_can_lower_moment_matched_score(self):
        """Increasing exact fixed-q Q need not increase the Gaussian score.

        z ~ N(0, 1), x = exp(-z**2 / 2) + epsilon, observed x=0.
        Symmetry makes Cov(z, x)=0, so the moment-matched posterior is
        unchanged. With C, d, W fixed, the upstream variance M-step is exact.
        """
        prior = pdf.GaussianPDF(mu=jnp.zeros((1, 1)),
                                Sigma=jnp.ones((1, 1, 1)))
        model = LSEMObservationModel(Dx=1, Dz=1, Dk=1)
        model.C = jnp.array([[0.0, 1.0]])
        model.d = jnp.zeros(1)
        model.W = jnp.array([[0.0, 1.0]])
        second_moment = 1.0 / np.sqrt(3.0)
        r_before = 1.0 - second_moment
        model.Qx = jnp.array([[r_before]])
        model.update_observation_density()
        observed = jnp.zeros((1, 1))

        posterior = model.filtering(prior, observed)
        np.testing.assert_allclose(posterior.mu, prior.mu, atol=1e-12)
        np.testing.assert_allclose(posterior.Sigma, prior.Sigma, atol=1e-12)
        q_before = float(model.compute_Q_function(posterior, observed))
        score_before = float(model.get_data_density(prior).evaluate_ln(
            observed, element_wise=True)[0])

        smooth = {key: value[None]
                  for key, value in posterior.to_dict().items()}
        model.Qx = model._update_Qx(observed[None], smooth)
        np.testing.assert_allclose(model.Qx, [[second_moment]], atol=1e-12)
        model.update_observation_density()
        q_after = float(model.compute_Q_function(posterior, observed))
        score_after = float(model.get_data_density(prior).evaluate_ln(
            observed, element_wise=True)[0])

        def exact_q(r):
            return -0.5 * (np.log(2 * np.pi * r) + second_moment / r)

        def moment_matched_score(r):
            predictive_variance = r + second_moment - 0.5
            return -0.5 * (np.log(2 * np.pi * predictive_variance)
                           + 0.5 / predictive_variance)

        np.testing.assert_allclose([q_before, q_after],
                                   [exact_q(r_before), exact_q(second_moment)],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            [score_before, score_after],
            [moment_matched_score(r_before),
             moment_matched_score(second_moment)],
            atol=1e-12, rtol=1e-12)
        self.assertGreater(q_after - q_before, 0.02)
        self.assertLess(score_after - score_before, -0.01)


if __name__ == "__main__":
    unittest.main()
