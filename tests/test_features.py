"""CPU tests for Equation (1)–(2) feature extraction."""

from __future__ import annotations

import numpy as np

from xganet.data.features import extract_flow_features, minmax_fit, minmax_scale


def test_flow_feature_mapping() -> None:
    duration_ms = np.array([1000.0, 0.0])
    in_bytes = np.array([100.0, 50.0])
    out_bytes = np.array([100.0, 50.0])
    in_pkts = np.array([5.0, 1.0])
    out_pkts = np.array([5.0, 1.0])
    src_port = np.array([80.0, 443.0])
    dst_port = np.array([1234.0, 5678.0])
    protocol = np.array([6, 17])

    feats = extract_flow_features(
        duration_ms, in_bytes, out_bytes, in_pkts, out_pkts, src_port, dst_port, protocol
    )
    assert feats.shape == (2, 11)
    assert np.isclose(feats[0, 0], 1000.0)
    assert np.isclose(feats[0, 1], 200.0)
    assert np.isclose(feats[0, 2], 10.0)
    assert np.isclose(feats[0, 8], 10.0)
    assert np.isclose(feats[0, 9], 20.0)
    assert np.isclose(feats[0, 10], 100.0)
    assert feats[0, 5] == 1.0 and feats[0, 6] == 0.0 and feats[0, 7] == 0.0
    assert feats[1, 5] == 0.0 and feats[1, 6] == 1.0
    assert feats[1, 8] > 0.0


def test_minmax_leaves_protocol_onehot() -> None:
    rng = np.random.RandomState(0)
    raw = rng.rand(16, 11).astype(np.float32) * 100
    raw[:, 5:8] = np.eye(3, dtype=np.float32)[rng.randint(0, 3, size=16)]
    feat_min, feat_max = minmax_fit(raw)
    scaled = minmax_scale(raw, feat_min, feat_max)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(scaled[:, 5:8], raw[:, 5:8], atol=1e-6)
