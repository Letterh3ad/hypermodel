import numpy as np

from hypermodel.observer import Observer, train_observer
from hypermodel.probe import probe_auroc


def _data(seed=0, n=2000, d=96):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32) * rng.uniform(0.1, 10, d).astype(np.float32)
    y = X[:, 3] / X[:, 3].std() + 0.8 * np.tanh(X[:, 7]) + 0.5 * rng.standard_normal(n) > 0
    split = rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2])
    return X, y, split == "train", split == "val", split == "test"


def test_z_has_the_requested_width_and_carries_the_signal():
    X, y, train, val, test = _data()
    obs = train_observer(X, y, train, val, k=16, seed=0)
    z = obs.encode(X)
    assert z.shape == (len(y), 16) and z.dtype == np.float32
    mean, _ = probe_auroc(z, y, train | val, test, seeds=(0,))
    assert mean > 0.85


def test_same_seed_same_observer():
    X, y, train, val, _ = _data(1, n=600)
    a = train_observer(X, y, train, val, k=8, seed=4, max_epochs=20)
    b = train_observer(X, y, train, val, k=8, seed=4, max_epochs=20)
    np.testing.assert_array_equal(a.encode(X), b.encode(X))


def test_save_and_load_round_trip(tmp_path):
    X, y, train, val, _ = _data(2, n=600)
    obs = train_observer(X, y, train, val, k=8, seed=0, max_epochs=10)
    obs.save(tmp_path / "observer.pt")
    again = Observer.load(tmp_path / "observer.pt")
    np.testing.assert_allclose(again.encode(X), obs.encode(X), atol=1e-6)


def test_noise_gives_a_chance_level_z():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1500, 32)).astype(np.float32)
    y = rng.random(1500) < 0.4
    split = rng.choice(["train", "val", "test"], 1500, p=[0.6, 0.2, 0.2])
    train, val, test = split == "train", split == "val", split == "test"
    z = train_observer(X, y, train, val, k=8, seed=0).encode(X)
    mean, _ = probe_auroc(z, y, train | val, test, seeds=(0,))
    assert 0.4 < mean < 0.6
