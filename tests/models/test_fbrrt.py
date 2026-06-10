r"""Regression tests for the FBRRT-SMC estimators in diffusion_rl.models.on_policy.

These guard the fixes for the FBSDE backward pass against an analytically
solvable case:

    base drift f = 0,  a = 1,  reward r(x) = c * x   (1-D),

so under the sampled SDE  X_1 | X_t = x ~ N(x, 2(1-t))  and the value function
is exact in closed form

    V(x, t) = log E[exp(r(X_1)) | X_t = x] = c*x + c^2 * (1 - t),
    H(x, t) = exp(V(x, t)),     grad_x V = c   (constant).

Because V is linear, the BSDE driver is exact (no discretization error), so the
FBRRT targets are *exactly* unbiased.  At the t=0 generation every particle sits
at the origin, so the mean of exp(target) over particles must equal
H(0, 0) = exp(c^2).  This gives a tight, low-variance regression check.

Covered fixes (see on_policy.py):
  (B) entropy weights moved off the target (uniform child mean) -> targets are
      independent of `entropy_lambda`.
  (D) ancestor-aligned multi-step + multinomial resampling -> `td_lambda` is
      unbiased for every lambda.
  (A,C) corrected control-variate driver + 1/sqrt(2a) Malliavin scaling ->
      `fbrrt_cv` is unbiased when v_policy == v_target and stays close when they
      differ.
  weights: FBRRT returns local-entropy LSMC regression weights (mean 1), and
      the dataset/training loop plumb them through.
"""

import inspect
import math

import torch

from diffusion_rl.models.on_policy import (
    OnPolicySMCDataset,
    OnPolicyValue,
    fbrrt_smc_grad_control,
    fbrrt_smc_grad_control_td_lambda,
    fbrrt_smc_grad_control_variate,
)

C = 0.5
A = 1.0
H00 = math.exp(C**2)  # true H(0,0) = E[exp(c * X_1) | X_0 = 0]
DEVICE = torch.device("cpu")


def base_drift(x, t):
    return torch.zeros_like(x)


def reward(x):
    return C * x.squeeze(-1)


def value(x, t):
    # Exact log-value V(x,t) = c*x + c^2 (1 - t).
    t = t.reshape(-1)
    xs = x.squeeze(-1)
    if t.numel() == 1:
        t = t.expand(xs.shape[0])
    return C * xs + C**2 * (1.0 - t)


def _t0_mean(samples):
    """Mean of exp(target) over the t=0 generation (all particles at x=0)."""
    mask = samples.t == 0.0
    assert mask.any(), "no t=0 generation in FBRRT output"
    return torch.exp(samples.v_hat[mask]).mean().item()


# ---------------------------------------------------------------------------
# (B) entropy weighting must NOT bias the target
# ---------------------------------------------------------------------------
def test_grad_control_target_unbiased_and_entropy_independent():
    """grad_control t=0 target equals H(0,0) for every entropy_lambda.

    Regression for the bug where the bootstrap target used an entropy-WEIGHTED
    child mean (`exp(v/lambda)`), biasing it upward as entropy_lambda shrank.
    The target is now the unweighted child mean, so it is entropy-independent.
    """
    means = {}
    for ent in (float("inf"), 1.0, 0.3):
        torch.manual_seed(0)
        s = fbrrt_smc_grad_control(
            a=A, n_steps=5, n_particles=512, branch=8, f=base_drift,
            v_theta=value, reward=reward, d=1, alpha=1.0,
            entropy_lambda=ent, device=DEVICE,
        )
        m = _t0_mean(s)
        means[ent] = m
        assert abs(m - H00) / H00 < 0.03, f"ent={ent}: {m} vs {H00}"
    # The target must be identical across entropy_lambda (same seed => same x=0
    # children; only resampling, which does not touch t=0 targets, differs).
    assert abs(means[1.0] - means[float("inf")]) / H00 < 1e-4
    assert abs(means[0.3] - means[float("inf")]) / H00 < 1e-4


# ---------------------------------------------------------------------------
# (D) td_lambda multi-step must be unbiased for every lambda
# ---------------------------------------------------------------------------
def test_td_lambda_unbiased_all_lambda():
    """td_lambda t=0 target equals H(0,0) for lambda in {0, 0.5, 1}.

    Regression for the GAE/resampling mis-alignment: the multi-step return was
    combined with positionally-mismatched parents, biasing the target by an
    amount that grew with lambda and with n_steps.  The fix gathers the
    downstream return back to each parent by ancestry and uses multinomial
    resampling so descendants are i.i.d.
    """
    for leff in (0.0, 0.5, 1.0):
        torch.manual_seed(0)
        s = fbrrt_smc_grad_control_td_lambda(
            a=A, n_steps=6, n_particles=512, branch=8, f=base_drift,
            v_theta=value, reward=reward, d=1, lambda_eff=leff, alpha=1.0,
            entropy_lambda=float("inf"), device=DEVICE,
        )
        m = _t0_mean(s)
        assert abs(m - H00) / H00 < 0.05, f"lambda_eff={leff}: {m} vs {H00}"


def test_td_lambda_defaults_to_multinomial_resampling():
    """The ancestor-aligned multi-step is only valid under multinomial
    resampling (systematic returns a near-diagonal index map that breaks the
    i.i.d.-descendant assumption)."""
    sig = inspect.signature(fbrrt_smc_grad_control_td_lambda)
    assert sig.parameters["resample_method"].default == "multinomial"


# ---------------------------------------------------------------------------
# (A,C) control-variate driver + Malliavin scaling
# ---------------------------------------------------------------------------
def test_cv_unbiased_when_policy_equals_target():
    """fbrrt_cv with v_policy == v_target reduces to grad_control: exact."""
    torch.manual_seed(0)
    s = fbrrt_smc_grad_control_variate(
        a=A, n_steps=5, n_particles=512, branch=8, f=base_drift,
        v_policy=value, v_target=value, reward=reward, d=1, alpha=1.0,
        entropy_lambda=float("inf"), device=DEVICE,
    )
    m = _t0_mean(s)
    assert abs(m - H00) / H00 < 0.03, f"{m} vs {H00}"


def test_cv_driver_robust_to_policy_target_mismatch():
    """fbrrt_cv stays near-unbiased when v_policy != v_target.

    A correct residual control variate recovers grad_x V_target regardless of
    v_policy, so the t=0 target must still be ~H(0,0).  Regression for the old
    driver (`-|z|^2 + 2(1-alpha) z.grad`) and the missing 1/sqrt(2a) Malliavin
    factor, which together biased the target by O(|v_target - v_policy|) -- here
    that buggy bias was tens of percent.  A large branch suppresses the residual
    control-variate variance so the deterministic fix is what is being checked.
    """
    # v_policy = V + perturbation with a constant (non-zero) extra gradient.
    def v_policy(x, t):
        return value(x, t) + 0.5 * x.squeeze(-1)

    torch.manual_seed(0)
    s = fbrrt_smc_grad_control_variate(
        a=A, n_steps=4, n_particles=512, branch=64, f=base_drift,
        v_policy=v_policy, v_target=value, reward=reward, d=1, alpha=1.0,
        entropy_lambda=float("inf"), device=DEVICE,
    )
    m = _t0_mean(s)
    assert abs(m - H00) / H00 < 0.10, f"cv biased under policy/target gap: {m} vs {H00}"


# ---------------------------------------------------------------------------
# regression weights and dataset/training plumbing
# ---------------------------------------------------------------------------
def test_entropy_regression_weights_shape_and_normalisation():
    """FBRRT returns per-sample regression weights (mean ~1); non-uniform for
    finite entropy_lambda, uniform for entropy_lambda=inf."""
    torch.manual_seed(0)
    s_fin = fbrrt_smc_grad_control(
        a=A, n_steps=5, n_particles=256, branch=8, f=base_drift, v_theta=value,
        reward=reward, d=1, alpha=1.0, entropy_lambda=1.0, device=DEVICE,
    )
    assert s_fin.weights.shape == s_fin.v_hat.shape
    assert math.isclose(s_fin.weights.mean().item(), 1.0, rel_tol=0.05)
    assert s_fin.weights.std().item() > 0.0  # genuinely reweighted

    torch.manual_seed(0)
    s_inf = fbrrt_smc_grad_control(
        a=A, n_steps=5, n_particles=256, branch=8, f=base_drift, v_theta=value,
        reward=reward, d=1, alpha=1.0, entropy_lambda=float("inf"), device=DEVICE,
    )
    assert torch.allclose(s_inf.weights, torch.ones_like(s_inf.weights))


def test_dataset_yields_weighted_tuples_and_training_step_runs():
    """The dataset yields (y, x, t, w); FBRRT weights are non-uniform, non-FBRRT
    weights are uniform, and the weighted training_step produces a finite loss."""

    def reward2(x):  # dim-agnostic reward returning shape (B,)
        return C * x.sum(dim=-1)

    class LinearVM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(3, 1)  # [x (2-d), t (1-d)]

        def forward(self, x, t):
            t = t.reshape(-1, 1) if t.ndim == 1 else t
            if t.shape[0] != x.shape[0]:
                t = t.expand(x.shape[0], 1)
            return self.net(torch.cat([x, t], dim=-1)).squeeze(-1)

    for method, expect_uniform in [
        ("fbrrt", False),
        ("single_seed_mc", True),
    ]:
        vm = LinearVM()
        ds = OnPolicySMCDataset(
            dim=2, drift=base_drift, value=vm, smc_value=vm, reward=reward2,
            device=DEVICE, sampling_method=method, a=A, batch_size=32,
            n_steps=6, mc_samples_per_step=8, branch=4, entropy_lambda=1.0,
        )
        it = iter(ds)
        rows = [next(it) for _ in range(16)]
        assert all(len(r) == 4 for r in rows), f"{method}: not a 4-tuple"
        batch = [torch.stack([r[i] for r in rows]) for i in range(4)]
        y, x, t, w = batch
        assert y.shape == (16, 1) and x.shape == (16, 2)
        assert t.shape == (16, 1) and w.shape == (16, 1)
        if expect_uniform:
            assert torch.allclose(w, torch.ones_like(w)), f"{method}: weights not uniform"
        else:
            assert w.std().item() > 0.0, f"{method}: weights unexpectedly uniform"

        model = OnPolicyValue(
            base_score_module=base_drift, value_module=vm, a=A, lr=1e-3, dim=2
        )
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss).all(), f"{method}: non-finite weighted loss"

    # Back-compat: training_step must still accept a legacy 3-tuple.
    vm = LinearVM()
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=vm, a=A, lr=1e-3, dim=2
    )
    y = torch.zeros(8, 1)
    x = torch.randn(8, 2)
    t = torch.rand(8, 1)
    loss3 = model.training_step((y, x, t), 0)
    assert torch.isfinite(loss3).all()
