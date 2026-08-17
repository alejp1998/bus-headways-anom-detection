import numpy as np
from scipy.spatial.distance import mahalanobis


def vectorized_mahalanobis(X, mean, iv):
    """Reference implementation of the vectorized distance used in detect_anoms_hws."""
    diff = X - mean
    return np.sqrt(np.maximum(0.0, np.sum((diff @ iv) * diff, axis=1)))


def test_vectorized_mahalanobis_matches_scipy():
    """The vectorized distance must be numerically identical to scipy's mahalanobis."""
    rng = np.random.default_rng(42)
    mean = np.array([300.0, 320.0])
    cov = np.array([[2500.0, 800.0], [800.0, 3000.0]])
    iv = np.linalg.inv(cov)
    X = rng.multivariate_normal(mean, cov, size=50)

    scipy_dists = np.array([mahalanobis(mean, row, iv) for row in X])
    numpy_dists = vectorized_mahalanobis(X, mean, iv)

    assert np.allclose(scipy_dists, numpy_dists, atol=1e-10)


def test_vectorized_mahalanobis_1d():
    """1D case: normalized absolute deviation equals |x - mean| / std."""
    mean = 300.0
    std = 50.0
    X = np.array([[200.0], [300.0], [400.0]])

    expected = np.abs(X[:, 0] - mean) / std
    actual = vectorized_mahalanobis(X, np.array([mean]), np.array([[1.0 / (std**2)]]))

    assert np.allclose(actual, expected)


def test_vectorized_mahalanobis_speed():
    """Bulk computation must be fast (vectorized, no row-wise Python loops)."""
    rng = np.random.default_rng(7)
    mean = rng.normal(300, 50, size=8)
    cov = np.eye(8) * 2000
    iv = np.linalg.inv(cov)
    X = rng.multivariate_normal(mean, cov, size=10_000)

    import time

    start = time.perf_counter()
    _ = vectorized_mahalanobis(X, mean, iv)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"Vectorized computation too slow: {elapsed:.3f}s"
