"""Comparison of NaDS and the baselines on artificial LFR benchmark graphs."""

import argparse
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
from _common import OUTPUTS

from nads import (
    CDS_solver,
    DEFAULT_PARAMS,
    NS_solver,
    Influence_evaluation,
    SingleDiscount,
    get_influencers,
    greedy,
    katz_seeds,
    kcore_seeds,
    set_seeds,
    weighted_graph,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", nargs="+", type=int, default=[1000, 2000, 5000])
    parser.add_argument("--mus", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--k-list", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--time-per-k", type=int, default=20, help="seconds of search budget per unit of K")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--buffer-dim", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUTPUTS / "lfr.pickle")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds()
    params = DEFAULT_PARAMS
    h0 = params[1]
    delta, xi, d, buffer_dim = args.delta, args.xi, args.d, args.buffer_dim
    results = {}

    for n in args.n_nodes:
        results[n] = {}
        for mu in args.mus:
            results[n][mu] = {}

            g = nx.LFR_benchmark_graph(n=n, tau1=3, tau2=1.5, mu=mu, average_degree=10, seed=42)
            g = g.to_undirected()
            g.remove_edges_from(nx.selfloop_edges(g))
            g, W = weighted_graph(g)

            for K in args.k_list:
                results[n][mu][K] = {}
                max_time = args.time_per_k * K
                print(K)

                start = time.time()
                x0 = katz_seeds(g, K, h0)
                results[n][mu][K]['katz'] = (Influence_evaluation(g, W, x0, params)[0], time.time() - start)
                print('katz:', results[n][mu][K]['katz'][0])

                start = time.time()
                x0 = kcore_seeds(g, K, h0)
                results[n][mu][K]['kcore'] = (Influence_evaluation(g, W, x0, params)[0], time.time() - start)
                print('kcore:', results[n][mu][K]['kcore'][0])

                start = time.time()
                influencers = get_influencers(g, d=3, max_influencers=K)
                x0 = np.zeros(len(g.nodes))
                x0[influencers] = h0
                results[n][mu][K]['CI'] = (Influence_evaluation(g, W, x0, params)[0], time.time() - start)
                print('CI:', results[n][mu][K]['CI'][0])

                start = time.time()
                results[n][mu][K]['greed'] = (greedy(g, W, params, K)[0], time.time() - start)
                print('greed:', results[n][mu][K]['greed'])

                start = time.time()
                s, x0 = SingleDiscount(g, W, params, K)
                results[n][mu][K]['disc'] = (s, time.time() - start)
                print('disc:', results[n][mu][K]['disc'][0])

                start = time.time()
                s, X, history = NS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim)
                results[n][mu][K]['ns'] = (s[-1], time.time() - start, history)

                start = time.time()
                s, X, history = CDS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim)
                results[n][mu][K]['cds'] = (s[-1], time.time() - start, history)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
