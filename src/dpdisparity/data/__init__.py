from .adult import load_adult
from .mnist import load_mnist


def load(dataset: str, seed: int) -> dict:
    return {"mnist": load_mnist, "adult": load_adult}[dataset](seed)
