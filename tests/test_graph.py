"""CPU tests for Algorithm 1 IP-OR-time edges and Equation (9) Â."""

from __future__ import annotations

import numpy as np
import torch

from xganet.data.graph import (
    batch_adjacency,
    encode_ip_ids,
    ip_degrees,
    normalized_adjacency,
    structure_features,
    temporal_neighbor_counts,
)


def test_ip_and_temporal_edges() -> None:
    src = torch.tensor([1, 1, 9, 3])
    dst = torch.tensor([2, 8, 2, 7])
    times = torch.tensor([0, 1000, 5, 50])
    adj = batch_adjacency(src, dst, times, delta_t=10)

    # 0 and 1 share src IP
    assert adj[0, 1] == 1.0
    # 0 and 2 share an endpoint (dst_0 == dst? 2 vs 2: node2 dst=2, node0 dst=2 — same dst)
    assert adj[0, 2] == 1.0
    # 0 and 2 also src_2=9 vs ... node0 src=1 dst=2; node2 src=9 dst=2 share dst
    # 2 and 0 already covered
    # 0 and 3: IPs disjoint, |0-50|=50 > 10 -> no edge
    assert adj[0, 3] == 0.0
    # 2 and 3: IPs disjoint, |5-50|=45 > 10 -> no edge
    assert adj[2, 3] == 0.0
    # diagonal is zero
    assert torch.count_nonzero(adj.diag()) == 0
    # undirected
    assert torch.allclose(adj, adj.T)


def test_eq4_cross_endpoint_edge() -> None:
    src = torch.tensor([10, 99])
    dst = torch.tensor([5, 10])
    times = torch.tensor([0, 10_000])
    adj = batch_adjacency(src, dst, times, delta_t=1)
    assert adj[0, 1] == 1.0


def test_normalized_adjacency_symmetric_with_self_loops() -> None:
    src = torch.tensor([1, 1, 2])
    dst = torch.tensor([3, 4, 5])
    times = torch.tensor([0, 1, 1000])
    adj = batch_adjacency(src, dst, times, delta_t=0)
    hat = normalized_adjacency(adj)
    assert torch.allclose(hat, hat.T, atol=1e-6)
    assert hat.shape == (3, 3)
    assert torch.all(hat.diag() > 0)


def test_ip_degrees_and_temporal_counts() -> None:
    src_ids = np.array([1, 1, 2])
    dst_ids = np.array([3, 4, 5])
    times = np.array([0, 3, 100])
    degrees = ip_degrees(src_ids, dst_ids)
    assert degrees[0] == 1.0
    assert degrees[1] == 1.0
    assert degrees[2] == 0.0
    temporal = temporal_neighbor_counts(times, delta_t=5)
    assert temporal[0] == 1.0
    assert temporal[1] == 1.0
    assert temporal[2] == 0.0
    struct = structure_features(src_ids, dst_ids, times, delta_t=5)
    assert struct.shape == (3, 4)


def test_encode_ip_ids_shared_vocab() -> None:
    src_ids, dst_ids, vocab = encode_ip_ids(
        np.array(["10.0.0.1", "10.0.0.2"]),
        np.array(["10.0.0.2", "10.0.0.3"]),
    )
    assert vocab["10.0.0.2"] == src_ids[1] == dst_ids[0]
    assert len(vocab) == 3
