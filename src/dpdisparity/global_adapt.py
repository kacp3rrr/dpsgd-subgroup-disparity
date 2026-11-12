"""DPSGD-Global-Adapt (Esipova et al., 2023, Sec. 5 / Alg. 2) as an Opacus DPOptimizer.

Per step, with C the clipping norm and Z a private, adaptive upper bound on gradient norms:
  1. scale each per-sample gradient by C/Z if its norm <= Z, otherwise clip it to C;
  2. sum, add N(0, sigma^2 C^2), average, and step (inherited from DPOptimizer);
  3. release a noisy count b = #{||g_i|| > tau * Z} + N(0, sigma2^2) and set Z <- Z * exp(-eta_Z + b / |B|).
The count is a sampled Gaussian mechanism (sensitivity 1, noise sigma2, rate q); `train.py` composes
`count_releases` of them into the same RDP accountant as the gradient steps.
"""

import math

import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed


def scale_factors(norms: torch.Tensor, C: float, Z: float) -> torch.Tensor:
    """C/Z below Z keeps gradients' relative sizes (and the update direction); above Z, clip to C."""
    return torch.where(norms <= Z, torch.full_like(norms, C / Z), C / (norms + 1e-6))


class GlobalAdaptOptimizer(DPOptimizer):
    def __init__(self, optimizer, *, Z0: float, tau: float, sigma2: float, eta_Z: float, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.Z, self.tau, self.sigma2, self.eta_Z = Z0, tau, sigma2, eta_Z
        self.count_releases = 0
        self._norms = None

    def clip_and_accumulate(self):
        if len(self.grad_samples[0]) == 0:  # empty Poisson batch: let Opacus handle it
            self._norms = None
            return super().clip_and_accumulate()
        norms = torch.stack([g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples], dim=1).norm(2, dim=1)
        factors = scale_factors(norms, self.max_grad_norm, self.Z)
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            g = self._get_flat_grad_sample(p).to(p.dtype)
            grad = torch.einsum("i,i...", factors.to(g.device, p.dtype), g)
            p.summed_grad = grad if p.summed_grad is None else p.summed_grad + grad
            _mark_as_processed(p.grad_sample)
        self._norms = norms

    def step(self, closure=None):
        loss = super().step(closure)
        if self._norms is not None:
            self._update_Z()
        return loss

    def _update_Z(self):
        # Uses the Z that scaled this batch, after the parameter step (as in dp-testing / Alg. 2).
        # Divides by the realized batch size; whether fair-dp uses the expected size is an open check.
        count = (self._norms > self.tau * self.Z).sum().item()
        noise = torch.normal(0.0, self.sigma2, (1,), generator=self.generator).item()
        self.Z *= math.exp(-self.eta_Z + (count + noise) / len(self._norms))
        self.count_releases += 1
        self._norms = None
