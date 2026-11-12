"""Per-group test metrics. Rates are fractions; a zero denominator gives None."""

import torch
import torch.nn.functional as F


def binary_rates(y: torch.Tensor, pred: torch.Tensor) -> dict:
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tp = int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())
    div = lambda a, b: a / b if b else None
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fnr": div(fn, fn + tp), "fpr": div(fp, fp + tn), "tpr": div(tp, fn + tp)}


@torch.no_grad()
def evaluate(model, X, y, g, group_names: dict, device="cpu", batch_size=1024) -> dict:
    """{"overall": {...}, "groups": {name: {n, accuracy, loss[, tn, fp, fn, tp, fnr, fpr, tpr]}}}."""
    model.eval()
    logits = torch.cat([model(X[i:i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)])
    losses = F.cross_entropy(logits, y, reduction="none")
    pred = logits.argmax(1)
    binary = logits.shape[1] == 2

    def stats(mask):
        out = {"n": int(mask.sum()), "accuracy": float((pred[mask] == y[mask]).float().mean()),
               "loss": float(losses[mask].mean())}
        return out | binary_rates(y[mask], pred[mask]) if binary else out

    return {"overall": stats(torch.ones_like(y, dtype=torch.bool)),
            "groups": {name: stats(g == k) for k, name in group_names.items()}}
