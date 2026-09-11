"""Tests for stratified class allocation used in NF-BoT-IoT-v2 subsampling."""

from __future__ import annotations

import numpy as np

from xganet.data.load_nf import allocate_class_counts, stratified_sample_indices


def test_keep_all_theft_and_floor() -> None:
    counts = {
        "DDoS": 1000,
        "DoS": 800,
        "Reconnaissance": 200,
        "Benign": 150,
        "Theft": 40,
    }
    alloc = allocate_class_counts(counts, sample_size=300, keep_all=("Theft",), min_per_class=20)
    assert alloc["Theft"] == 40
    assert sum(alloc.values()) == 300
    for name, take in alloc.items():
        assert 0 <= take <= counts[name]


def test_stratified_indices_reproducible() -> None:
    attacks = np.array(["DDoS"] * 50 + ["DoS"] * 40 + ["Theft"] * 10)
    a = stratified_sample_indices(attacks, 40, seed=1, keep_all=("Theft",), min_per_class=5)
    b = stratified_sample_indices(attacks, 40, seed=1, keep_all=("Theft",), min_per_class=5)
    c = stratified_sample_indices(attacks, 40, seed=2, keep_all=("Theft",), min_per_class=5)
    np.testing.assert_array_equal(a, b)
    assert len(a) <= 40
    assert not np.array_equal(a, c)
    theft_idx = np.flatnonzero(attacks == "Theft")
    assert set(theft_idx).issubset(set(a.tolist()))
