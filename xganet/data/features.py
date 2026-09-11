"""Equation (1)–(2) flow features for the X-GANet feature branch."""

from __future__ import annotations

import numpy as np

TCP_PROTO = 6
UDP_PROTO = 17
ICMP_PROTO = 1
EPS = 1e-6

FEATURE_COLUMNS = (
    "duration",
    "bytes",
    "packets",
    "src_port",
    "dst_port",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "flow_rate",
    "avg_pkt_size",
    "iat",
)


def extract_flow_features(
    duration_ms: np.ndarray,
    in_bytes: np.ndarray,
    out_bytes: np.ndarray,
    in_pkts: np.ndarray,
    out_pkts: np.ndarray,
    src_port: np.ndarray,
    dst_port: np.ndarray,
    protocol: np.ndarray,
) -> np.ndarray:
    """Build the d-dimensional flow vector from Equation (1).

    Returns an unscaled float32 array of shape (N, 11).
    """
    duration_ms = duration_ms.astype(np.float64, copy=False)
    in_bytes = in_bytes.astype(np.float64, copy=False)
    out_bytes = out_bytes.astype(np.float64, copy=False)
    in_pkts = in_pkts.astype(np.float64, copy=False)
    out_pkts = out_pkts.astype(np.float64, copy=False)
    src_port = src_port.astype(np.float64, copy=False)
    dst_port = dst_port.astype(np.float64, copy=False)
    protocol = protocol.astype(np.int64, copy=False)

    duration = duration_ms
    total_bytes = in_bytes + out_bytes
    total_pkts = in_pkts + out_pkts
    duration_s = np.maximum(duration / 1000.0, EPS)
    pkt_safe = np.maximum(total_pkts, 1.0)
    flow_rate = total_pkts / duration_s
    avg_pkt_size = total_bytes / pkt_safe
    iat = duration / pkt_safe

    proto_tcp = (protocol == TCP_PROTO).astype(np.float64)
    proto_udp = (protocol == UDP_PROTO).astype(np.float64)
    proto_icmp = (protocol == ICMP_PROTO).astype(np.float64)

    features = np.stack(
        [
            duration,
            total_bytes,
            total_pkts,
            src_port,
            dst_port,
            proto_tcp,
            proto_udp,
            proto_icmp,
            flow_rate,
            avg_pkt_size,
            iat,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def minmax_fit(features: np.ndarray, onehot_slice: slice = slice(5, 8)) -> tuple[np.ndarray, np.ndarray]:
    """Fit min–max statistics. One-hot protocol columns are left unscaled."""
    feat_min = features.min(axis=0).astype(np.float32)
    feat_max = features.max(axis=0).astype(np.float32)
    feat_min[onehot_slice] = 0.0
    feat_max[onehot_slice] = 1.0
    return feat_min, feat_max


def minmax_scale(
    features: np.ndarray,
    feat_min: np.ndarray,
    feat_max: np.ndarray,
) -> np.ndarray:
    """Equation (2) min–max scaling with a floor on the denominator."""
    denom = np.maximum(feat_max - feat_min, EPS)
    scaled = (features - feat_min) / denom
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)
