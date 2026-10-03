# What EM monotonicity means for LSEM models

LSEM observation and state models use Gaussian moment matching for filtering,
prediction, and smoothing. Their predictive log likelihood is the score of the
resulting Gaussian predictive distributions. It is generally an approximation
to the marginal log likelihood of the nonlinear generative model.

The M-step uses a different quantity: the expected complete-data log likelihood
under a fixed smoothing distribution,

\[
Q(\theta;q)=\mathbb E_q[\log p_\theta(x,z)].
\]

For LSEM features and Gaussian smoothing distributions, the integrals in this
quantity can be calculated analytically. An exact calculation of this expectation
does not make the Gaussian smoothing distribution an exact nonlinear posterior.

Exact EM obtains its likelihood guarantee by using
\(q=p_{\theta_{\rm old}}(z\mid x)\). Increasing Q for that fixed posterior,
even through partial coordinate updates, increases or preserves the true
marginal likelihood. Gaussian moment matching does not supply that posterior.
Consequently, neither the exact-EM guarantee for the true likelihood nor a
monotonicity guarantee for the Gaussian predictive score follows.

## A deterministic counterexample

Consider a one-dimensional LSEM observation model with a fixed Gaussian prior:

\[
z\sim N(0,1),\qquad k(z)=e^{-z^2/2},\qquad
x=k(z)+\epsilon,\quad \epsilon\sim N(0,r),\qquad x_{\rm observed}=0.
\]

Hold the loading, offset and kernel weights fixed and update only the observation
noise variance. This is a valid coordinate M-step, so exact generalized EM would
still have its monotonicity guarantee if the E-step used the exact posterior.

Write

\[
m=\mathbb E[k(z)]=\frac1{\sqrt2},\qquad
s=\mathbb E[k(z)^2]=\frac1{\sqrt3},\qquad
v=\operatorname{Var}(k(z))=s-\frac12.
\]

Symmetry gives \(\operatorname{Cov}(z,x)=0\). Conditioning the moment-matched
joint Gaussian on the observation therefore leaves the approximate posterior
unchanged: \(q(z)=N(0,1)\). The true nonlinear posterior is not this Gaussian.

The exact fixed-q observation objective and the Gaussian predictive score are

\[
Q(r)=-\frac12\left[\log(2\pi r)+\frac{s}{r}\right],
\qquad
\ell_{\rm MM}(r)=-\frac12\left[
\log(2\pi(r+v))+\frac{m^2}{r+v}\right].
\]

Start at \(r_0=1-s\). This maximizes the Gaussian predictive score because
\(r_0+v=m^2=1/2\). The exact variance M-step instead sets
\(r_1=\mathbb E_q[(x-k(z))^2]=s\), which maximizes Q. Since \(r_1\ne r_0\),
the predictive score strictly decreases.

| Quantity | Before | After | Change |
|---|---:|---:|---:|
| Noise variance | 0.4226497308103742 | 0.5773502691896257 | +0.1547005383792515 |
| Fixed-q observation Q | -1.1713454838386466 | -1.1442854610376452 | +0.0270600228010014 |
| Gaussian predictive log score | -1.0723649429247000 | -1.0890038329970175 | -0.0166388900723176 |

The regression test uses the upstream `LSEMObservationModel.filtering`,
`_update_Qx`, `compute_Q_function`, and `get_data_density` implementations. It
checks the closed-form values in float64. No numerical optimization, missing-data
handling, initial-state update, or state-transition approximation is involved.
Thus an optimizer iteration limit or one of those other mechanisms is not
required to produce a decrease. This example demonstrates an inherent limitation;
it does not rule out additional implementation bugs in a particular application.

## What to monitor

1. **Fixed-posterior M-step progress.** Record Q immediately before and after each
   partial update, using exactly the same smoothing distribution. Include the
   prior and only observed entries as appropriate. If an update optimizes a
   regularized objective, compare that objective as well as unregularized Q.
   A finite optimizer iteration limit alone does not prevent ascent when its
   accepted steps improve the objective; failed line searches and nonfinite
   results need separate diagnostics.
2. **Predictive performance.** Continue reporting training and held-out Gaussian
   predictive scores, explicitly identifying them as approximate scores. Their
   decreases cannot, by themselves, establish an M-step bug. Conversely, Q ascent
   does not establish better predictive performance.
3. **A variational lower bound, if required.** For a normalized and consistent
   Gaussian Markov smoothing distribution, calculate
   \(\mathcal F(q,\theta)=Q(\theta;q)+H(q)\). This is a lower bound on the true
   nonlinear model's marginal log likelihood. For states \(z_0,\ldots,z_{T-1}\),
   its entropy can be computed from adjacent-pair and singleton marginals:
   \[
   H(q)=\sum_{t=0}^{T-2}H(q(z_t,z_{t+1}))
         -\sum_{t=1}^{T-2}H(q(z_t)).
   \]
   For one state, use its singleton entropy. The Gaussian moment-matching E-step
   is not a variational optimization of this bound and need not increase it.

Raw Q values from successive EM iterations are not comparable as optimization
progress because the smoothing distribution changes. Fixed-q Q differences
diagnose the M-step; a bound that includes entropy and a predictive score answer
different questions.

An implementation error that decreases the intended M-step objective should be
fixed independently. Correcting such an error does not turn moment-matched EM
into exact EM or supply a predictive-score monotonicity guarantee.

## Mixed observation and model precision

The masked LSEM observation M-step expands the expected squared residual into
`sum(x**2) - 2*sum(x*E[prediction]) + sum(E[prediction**2])`.
If observations are stored in float32 and moments and parameters are float64,
the first sum must be promoted before squaring and summation. Otherwise its
rounding error remains in the residual even though the other terms are float64.
This matters when observation noise is small compared with the signal.

`MaskedLSEMObservationModel._masked_quadratic` promotes observations to the
common dtype of the observations, moments and loadings. This fixes a separate
M-step bug; it leaves all-float32 computation in float32. The regression freezes
the posterior and kernel weights, starts at the noise optimum, and checks the
updated variance and Q against an independently evaluated expected residual.
On the offset-10 fixture in one CPU reproduction, the previous float32 sum forced a variance near
`1e-5` down to the `1e-6` floor, decreasing true Q by about 2936.77 nats.
After promotion, the variance stays at the optimum and Q is preserved to
floating-point precision. The analytic counterexample above still applies.
