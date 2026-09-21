"""NaDS vs CDS and the baselines on a real-world network, for several seed-set sizes K.

For each K this runs Katz, single discount, RIS and greedy, then NaDS (NS) and
CDS started from the single-discount seeds ("disc"), and finally 10 runs of NS
and CDS from random starting sets drawn from the 4K highest-degree nodes.
Each solver gets K * time-per-k seconds.

Output layout (compatible with the published results/ folder):

    <out>/<network>/baselines.csv                  method, K, score, time
    <out>/<network>/ns_cds_disc/{ns,cds}_disc_<K>.csv   solver history rows
    <out>/<network>/ns_cds_random/{ns,cds}_<K>.csv      best-so-far curve per run
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from _common import DATA, OUTPUTS

from nads import (
    CDS_solver,
    DEFAULT_PARAMS,
    NETWORK_FILES,
    NS_solver,
    RIS,
    Influence_evaluation,
    SingleDiscount,
    best_so_far_curve,
    greedy,
    katz_seeds,
    load_network,
    set_seeds,
    weighted_graph,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", required=True, choices=list(NETWORK_FILES))
    parser.add_argument("--k-list", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--time-per-k", type=int, default=200, help="seconds of search budget per unit of K")
    parser.add_argument("--random-runs", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--buffer-dim", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUTS)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds()
    params = DEFAULT_PARAMS
    h0 = params[1]
    delta, xi, d, buffer_dim = args.delta, args.xi, args.d, args.buffer_dim

    out = args.out / args.network
    (out / "ns_cds_disc").mkdir(parents=True, exist_ok=True)
    (out / "ns_cds_random").mkdir(parents=True, exist_ok=True)

    g, W = weighted_graph(load_network(args.network, args.data_dir))
    R = len(g.nodes)
    degrees = np.sum(W.toarray() > 0, axis=0)
    baselines = []

    for K in args.k_list:
        max_time = K * args.time_per_k
        print(K)

        start = time.time()
        x0 = katz_seeds(g, K, h0)
        baselines.append(("katz", K, Influence_evaluation(g, W, x0, params)[0], time.time() - start))
        print("katz:", baselines[-1][2])

        start = time.time()
        s, x0 = SingleDiscount(g, W, params, K)
        baselines.append(("disc", K, s, time.time() - start))
        print("disc:", s)

        start = time.time()
        baselines.append(("ris", K, RIS(g, W, params, K, R)[0], time.time() - start))
        print("ris:", baselines[-1][2])

        start = time.time()
        baselines.append(("greed", K, greedy(g, W, params, K)[0], time.time() - start))
        print("greed:", baselines[-1][2], baselines[-1][3])

        _, _, history = NS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim)
        np.savetxt(out / "ns_cds_disc" / f"ns_disc_{K}.csv", history)
        _, _, history = CDS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim)
        np.savetxt(out / "ns_cds_disc" / f"cds_disc_{K}.csv", history)

        top_nc = list(np.argsort(degrees)[-4 * K:])
        ns, cds = [], []
        for _ in range(args.random_runs):
            x0 = np.zeros(len(g.nodes))
            x0[list(random.sample(top_nc, K))] = h0
            print("rand_nc: " + str(Influence_evaluation(g, W, x0, params)[0]))

            _, _, history = NS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim)
            ns.append(best_so_far_curve(history, max_time))
            _, _, history = CDS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim)
            cds.append(best_so_far_curve(history, max_time))
        np.savetxt(out / "ns_cds_random" / f"ns_{K}.csv", ns)
        np.savetxt(out / "ns_cds_random" / f"cds_{K}.csv", cds)

    with open(out / "baselines.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "K", "score", "time"])
        writer.writerows(baselines)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
