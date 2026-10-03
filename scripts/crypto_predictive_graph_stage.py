"""Actual historical-neighbor graph candidate using CUDA kNN and Leiden communities.

This method was named in NEXT_PREDICTIVE_STATES_STAGE before the first test run.
Its implementation follows the centroid run; the 2026 comparison is therefore
reported as exploratory family analysis, never used to replace the locked winner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import igraph as ig
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_predictive_states_stage import (CFG, SCOPE, load_panel, period_indices,
                                                      score_row, SYMBOLS)
from src.crypto.predictive_states import (TASK_SIZES, distribution, log_loss,
                                          make_representation, outcome_labels)


def actual_knn(query: np.ndarray, reference: np.ndarray, *, k: int = 12,
               query_symbol: np.ndarray | None = None, reference_symbol: np.ndarray | None = None,
               query_time: np.ndarray | None = None, reference_time: np.ndarray | None = None,
               batch: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest *observed* reference states; main distance matrix runs on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("This stage requires CUDA, as requested")
    ref = torch.as_tensor(reference, dtype=torch.float32, device="cuda")
    ref2 = (ref * ref).sum(axis=1)
    if reference_symbol is not None:
        ref_symbol = torch.as_tensor(reference_symbol.astype(np.int16), device="cuda")
        ref_time = torch.as_tensor(reference_time.astype(np.int64), device="cuda")
    ids = np.empty((len(query), k), dtype=np.int32)
    distance = np.empty((len(query), k), dtype=np.float32)
    for start in range(0, len(query), batch):
        end = min(len(query), start + batch)
        x = torch.as_tensor(query[start:end], dtype=torch.float32, device="cuda")
        d2 = torch.clamp((x * x).sum(axis=1, keepdim=True) + ref2 - 2 * x @ ref.T, min=0)
        if query_symbol is not None:
            sym = torch.as_tensor(query_symbol[start:end].astype(np.int16), device="cuda")
            ts = torch.as_tensor(query_time[start:end].astype(np.int64), device="cuda")
            near_same_symbol_time = (sym[:, None] == ref_symbol[None, :]) & (torch.abs(ts[:, None] - ref_time[None, :]) < 4 * 3600 * 10**9)
            d2.masked_fill_(near_same_symbol_time, float("inf"))
        value, pos = torch.topk(d2, k, dim=1, largest=False)
        ids[start:end] = pos.cpu().numpy()
        distance[start:end] = torch.sqrt(value).cpu().numpy()
    return ids, distance


def weighted_vote(neighbors: np.ndarray, distance: np.ndarray, community: np.ndarray,
                  n_communities: int) -> np.ndarray:
    votes = np.zeros((len(neighbors), n_communities), dtype=np.float32)
    weights = 1 / np.maximum(distance, .05)
    rows = np.repeat(np.arange(len(neighbors)), neighbors.shape[1])
    np.add.at(votes, (rows, community[neighbors].reshape(-1)), weights.reshape(-1))
    return votes.argmax(axis=1).astype(np.int16)


def predict_distribution(states: np.ndarray, symbols: np.ndarray, accepted: np.ndarray,
                         general: dict[int, list[np.ndarray]], local: dict[tuple[int, int], list[np.ndarray]],
                         background: dict[int, list[np.ndarray]]) -> list[np.ndarray]:
    out = [np.empty((len(states), classes), dtype=np.float32) for classes in TASK_SIZES]
    for state in np.unique(states):
        positions = np.flatnonzero(states == state)
        for symbol in np.unique(symbols[positions]):
            rows = positions[symbols[positions] == symbol]
            probabilities = local.get((int(state), int(symbol)), general[int(state)])
            for task in range(3):
                out[task][rows] = probabilities[task]
    for symbol in np.unique(symbols[~accepted]):
        rows = np.flatnonzero((symbols == symbol) & ~accepted)
        for task in range(3):
            out[task][rows] = background[int(symbol)][task]
    return out


def main() -> None:
    panel = load_panel()
    periods = period_indices(panel)
    labels = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(),
                            panel.future_rv4.to_numpy(), panel.rv_slow_15m.to_numpy(),
                            tuple(CFG["return_bin_edges"]))
    symbols = panel.symbol_code.to_numpy(dtype=np.int8)
    x = make_representation(panel, CFG)["X1_endpoints"]
    rng = np.random.default_rng(CFG["random_seed"] + 101)
    anchors = []
    for symbol in range(len(SYMBOLS)):
        choices = periods["fit"][symbols[periods["fit"]] == symbol]
        anchors.extend(rng.choice(choices, size=1000, replace=False))
    anchors = np.array(anchors, dtype=np.int64)
    time_ns = panel.available_at.astype("int64").to_numpy()
    neighbors, distances = actual_knn(x[anchors], x[anchors], k=16,
                                      query_symbol=symbols[anchors], reference_symbol=symbols[anchors],
                                      query_time=time_ns[anchors], reference_time=time_ns[anchors])
    edges = set()
    for row, near in enumerate(neighbors):
        edges.update((min(row, int(other)), max(row, int(other))) for other in near if row != other)
    graph = ig.Graph(n=len(anchors), edges=list(edges), directed=False)
    groups = graph.community_multilevel()
    community = np.asarray(groups.membership, dtype=np.int16)
    size = np.bincount(community)
    print(f"CUDA kNN graph {len(anchors)} actual nodes, {graph.ecount()} edges, {len(size)} communities; sizes={size.tolist()}", flush=True)
    if len(size) > 100:
        raise RuntimeError("Graph resolution produced too many communities; do not silently overfit")

    # All outcome fitting uses discovery samples. The graph uses only pre-2025 inputs.
    discovery = periods["discovery"]
    sample = np.sort(rng.choice(discovery, size=len(discovery) // 4, replace=False))
    fitted_near, fitted_dist = actual_knn(x[sample], x[anchors], k=8)
    fitted_state = weighted_vote(fitted_near, fitted_dist, community, len(size))
    base = distribution(labels[sample])
    background = {int(s): distribution(labels[sample[symbols[sample] == s]], base, CFG["symbol_shrinkage"])
                  for s in np.unique(symbols[sample])}
    general = {}
    local = {}
    radius = {}
    for state in range(len(size)):
        rows = sample[fitted_state == state]
        if not len(rows):
            continue
        general[state] = distribution(labels[rows], base, CFG["shrinkage"])
        radius[state] = float(np.quantile(fitted_dist[fitted_state == state, -1], .9))
        for sym in np.unique(symbols[rows]):
            local_rows = rows[symbols[rows] == sym]
            local[(state, int(sym))] = distribution(labels[local_rows], general[state], CFG["symbol_shrinkage"])

    results = []
    state_cards = []
    prediction_frames = []
    for period in ("validation", "test"):
        idx = periods[period]
        near, dist = actual_knn(x[idx], x[anchors], k=8)
        state = weighted_vote(near, dist, community, len(size))
        accepted = np.fromiter((size[int(s)] >= 100 and int(s) in radius and dist[j, -1] <= radius[int(s)]
                                for j, s in enumerate(state)), dtype=bool, count=len(state))
        # Rare graph communities without discovery support abstain.
        for missing_state in set(np.unique(state)) - set(general):
            general[int(missing_state)] = base
        p = predict_distribution(state, symbols[idx], accepted, general, local, background)
        results.append(score_row("real_neighbor_graph_X1", period, labels[idx], p,
                                 panel.available_at.iloc[idx], accepted, len(size)))
        source = panel.iloc[idx]
        market = source.groupby("available_at").return4.transform("mean").to_numpy()
        for cluster in np.unique(state):
            use = (state == cluster) & accepted
            if not use.any():
                continue
            outcome = source.return4.to_numpy()[use]
            state_cards.append({"period": period, "community": int(cluster), "rows": int(use.sum()),
                                "days": int(source.available_at.iloc[np.flatnonzero(use)].dt.floor("D").nunique()),
                                "mean_4h_bp": float(outcome.mean() * 10000),
                                "mean_4h_residual_bp": float((outcome - market[use]).mean() * 10000)})
        prediction_frames.append(pd.DataFrame({"symbol": source.symbol.to_numpy(),
                                               "available_at": source.available_at.to_numpy(),
                                               "period": period, "community": state, "accepted": accepted,
                                               "k8_distance": dist[:, -1],
                                               "logloss": log_loss(labels[idx], p).mean(axis=1)}))
        print(results[-1], flush=True)
    pd.DataFrame(results).to_csv(SCOPE / "neighbor_graph_scores.csv", index=False)
    pd.DataFrame(state_cards).to_csv(SCOPE / "neighbor_graph_cards.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(SCOPE / "neighbor_graph_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame({"anchor_row": anchors, "symbol": panel.symbol.iloc[anchors].to_numpy(),
                  "available_at": panel.available_at.iloc[anchors].to_numpy(),
                  "community": community, "knn16_distance": distances[:, -1]}).to_csv(SCOPE / "neighbor_graph_anchors.csv", index=False)
    joblib.dump({"anchors": anchors, "community": community, "radius": radius,
                 "general": general, "local": local, "background": background},
                SCOPE / "neighbor_graph_model.joblib", compress=3)
    (SCOPE / "neighbor_graph_manifest.json").write_text(json.dumps({
        "method": "CUDA exact kNN to 12,000 real history anchors; 16-neighbor graph; Louvain communities; 8-neighbor inference",
        "anchors_from": "2023-01 through 2024-12 only", "community_count": len(size), "community_sizes": size.tolist(),
        "validation_or_test_used_to_choose_graph": False,
        "caution": "implemented after first centroid test read; graph test comparison is exploratory"
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
