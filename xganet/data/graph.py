"""Algorithm 1 graph construction: IP-OR-time edges and symmetric Â."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def encode_ip_ids(src_ip: np.ndarray, dst_ip: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Map IP strings to dense integer identifiers shared across src/dst."""
    vocab: dict[str, int] = {}
    src_ids = np.empty(len(src_ip), dtype=np.int64)
    dst_ids = np.empty(len(dst_ip), dtype=np.int64)
    for i, ip in enumerate(src_ip):
        key = str(ip)
        ident = vocab.get(key)
        if ident is None:
            ident = len(vocab)
            vocab[key] = ident
        src_ids[i] = ident
    for i, ip in enumerate(dst_ip):
        key = str(ip)
        ident = vocab.get(key)
        if ident is None:
            ident = len(vocab)
            vocab[key] = ident
        dst_ids[i] = ident
    return src_ids, dst_ids, vocab


def ip_degrees(src_ids: np.ndarray, dst_ids: np.ndarray) -> np.ndarray:
    """Count other flows that share any endpoint IP with each flow (Eq. 4)."""
    ip_to_nodes: dict[int, set[int]] = defaultdict(set)
    n = len(src_ids)
    for i in range(n):
        ip_to_nodes[int(src_ids[i])].add(i)
        ip_to_nodes[int(dst_ids[i])].add(i)
    degrees = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = ip_to_nodes[int(src_ids[i])] | ip_to_nodes[int(dst_ids[i])]
        degrees[i] = float(len(neighbors) - 1)
    return degrees


def temporal_neighbor_counts(time_index: np.ndarray, delta_t: int) -> np.ndarray:
    """Count other flows with |t_i - t_j| <= delta_t using a sorted sweep."""
    n = len(time_index)
    counts = np.zeros(n, dtype=np.float32)
    if n == 0:
        return counts
    order = np.argsort(time_index, kind="mergesort")
    times = time_index[order].astype(np.int64, copy=False)
    left = 0
    for i in range(n):
        t_i = times[i]
        while t_i - times[left] > delta_t:
            left += 1
        right = i
        while right + 1 < n and times[right + 1] - t_i <= delta_t:
            right += 1
        counts[order[i]] = float(right - left)
    return counts


def structure_features(
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    time_index: np.ndarray,
    delta_t: int,
) -> np.ndarray:
    """Table 4 structure-branch attributes: degree, centrality, IP and time flags."""
    n = len(src_ids)
    degree = ip_degrees(src_ids, dst_ids)
    denom = max(n - 1, 1)
    centrality = degree / float(denom)
    ip_flag = (degree > 0).astype(np.float32)
    temporal = temporal_neighbor_counts(time_index, delta_t)
    time_flag = (temporal > 0).astype(np.float32)
    return np.stack([degree, centrality, ip_flag, time_flag], axis=1).astype(np.float32)


def batch_adjacency(
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    time_index: torch.Tensor,
    delta_t: int,
) -> torch.Tensor:
    """Induced IP-OR-time adjacency for a mini-batch (Algorithm 1 / Eq. 4–6).

    Returns a float tensor of shape (B, B) with zeros on the diagonal.
    """
    src = src_ids.unsqueeze(1)
    dst = dst_ids.unsqueeze(1)
    src_t = src_ids.unsqueeze(0)
    dst_t = dst_ids.unsqueeze(0)
    ip_adj = (src == src_t) | (dst == dst_t) | (src == dst_t) | (dst == src_t)
    time_adj = (time_index.unsqueeze(1) - time_index.unsqueeze(0)).abs() <= delta_t
    adj = (ip_adj | time_adj).to(dtype=torch.float32)
    adj.fill_diagonal_(0.0)
    return adj


def normalized_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Equation (9): Â = D^{-1/2} (A + I) D^{-1/2}."""
    batch = adj.size(0)
    identity = torch.eye(batch, device=adj.device, dtype=adj.dtype)
    a_tilde = adj + identity
    deg = a_tilde.sum(dim=-1).clamp(min=1.0)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt.unsqueeze(1) * a_tilde * inv_sqrt.unsqueeze(0)
