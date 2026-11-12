from .adult import load_adult
from .folktables import load_folktables
from .mnist import load_mnist


def load(dataset: str, seed: int, balance: bool | None = None, n_train: int | None = None) -> dict:
    """`folktables_sex` / `folktables_race` take the data options; MNIST and Adult have fixed splits."""
    if dataset.startswith("folktables_"):
        return load_folktables(seed, group=dataset.removeprefix("folktables_"), balance=bool(balance), n_train=n_train)
    assert balance is None and n_train is None, f"{dataset} has no data options"
    return {"mnist": load_mnist, "adult": load_adult}[dataset](seed)
