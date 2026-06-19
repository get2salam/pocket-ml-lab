"""Deterministic train/test split using a seeded Fisher-Yates shuffle."""

from typing import Any


def train_test_split(
    rows: list[dict[str, Any]],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train and test sets deterministically.

    Uses a seeded linear-congruential PRNG so results are reproducible
    across Python versions without importing random state from the stdlib.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows to split")

    indices = list(range(len(rows)))
    _lcg_shuffle(indices, seed)

    n_test = max(1, round(len(rows) * test_size))
    n_test = min(n_test, len(rows) - 1)

    test_indices = set(indices[:n_test])
    train = [row for i, row in enumerate(rows) if i not in test_indices]
    test = [row for i, row in enumerate(rows) if i in test_indices]
    return train, test


def _lcg_shuffle(items: list, seed: int) -> None:
    """In-place Fisher-Yates shuffle with a linear-congruential generator.

    LCG parameters are the same as glibc (multiplier=1103515245, increment=12345,
    modulus=2^31) for well-tested statistical properties.
    """
    state = seed & 0x7FFFFFFF
    m, a, c = 0x80000000, 1103515245, 12345

    n = len(items)
    for i in range(n - 1, 0, -1):
        state = (a * state + c) % m
        j = state % (i + 1)
        items[i], items[j] = items[j], items[i]
