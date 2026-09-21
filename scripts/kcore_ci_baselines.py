"""k-core and collective-influence (CI) baselines on the real-world networks."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from _common import DATA, OUTPUTS

from nads import (
    DEFAULT_PARAMS,
    NETWORK_FILES,
    Influence_evaluation,
    get_influencers,
    kcore_seeds,
    load_network,
    set_seeds,
    weighted_graph,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", nargs="+", default=list(NETWORK_FILES), choices=list(NETWORK_FILES))
    parser.add_argument("--k-list", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUTS / "kcore_ci.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds()
    params = DEFAULT_PARAMS
    h0 = params[1]
    rows = []

    for network in args.networks:
        print(f"========= {network} =========")
        g, W = weighted_graph(load_network(network, args.data_dir))

        for K in args.k_list:
            print(f"-------- K = {K} --------")

            start = time.time()
            x0 = kcore_seeds(g, K, h0)
            score = Influence_evaluation(g, W, x0, params)[0]
            rows.append((network, K, "kcore", score, time.time() - start))
            print(f"kcore:\t{rows[-1][3]:3f}\t{rows[-1][4]:3f}")

            start = time.time()
            influencers = get_influencers(g, d=3, max_influencers=K)
            x0 = np.zeros(len(g.nodes))
            x0[influencers] = h0
            score = Influence_evaluation(g, W, x0, params)[0]
            rows.append((network, K, "CI", score, time.time() - start))
            print(f"CI:\t{rows[-1][3]:3f}\t{rows[-1][4]:3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["network", "K", "method", "score", "time"])
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
