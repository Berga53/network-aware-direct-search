# NaDS: Network-aware Direct Search

Code and results for **NaDS**, a network-aware direct-search method for
influence maximization on (normal, non-hyper) graphs, published in the
*Journal of Complex Networks* (2025), [doi:10.1093/comnet/cnaf042](https://doi.org/10.1093/comnet/cnaf042). Given a graph, a seed-set size `K` and a
influence-propagation model, NaDS looks for the `K` seed nodes that
maximise the total influence spread. It searches over neighbours of the current
seeds in the network first, and falls back to a generic swap neighbourhood when
that stalls. The repository also contains the baselines it is compared with:
greedy, single discount, RIS, collective influence, k-core, Katz centrality,
random search, and the custom direct search (CDS).

Across the code and result files, `NS` stands for NaDS.

> The exact code that produced the published results is tagged
> `v1.0-paper` (a single notebook, `NaDS.ipynb`, with
> `csv/` and `Networks/` folders). This layout is a refactor of that code into
> a package; the algorithms are unchanged.

## Repository layout

- `src/nads/`: the importable Python package.
  - `models.py`: the influence-propagation model (`Influence_evaluation`) and the default parameters.
  - `direct_search.py`: `NS_solver` (NaDS) and `CDS_solver`. The `*_solver` variants stop after a time budget (used in the paper); the `*_solver_calls` variants stop after a number of influence evaluations.
  - `baselines.py`: greedy, single discount, RIS, collective influence, k-core and Katz seeds, random search, brute force.
  - `helpers.py`: graph construction, network loading, and small utilities.
- `NaDS.ipynb`: driver notebook with a short live demo and summaries of the published results.
- `scripts/`: the experiment runners.
  - `real_networks_comparison.py`: NaDS vs CDS and the baselines on one real network (one run per network).
  - `kcore_ci_baselines.py`: k-core and collective-influence baselines on all real networks.
  - `lfr_experiments.py`: comparison on artificial LFR graphs.
- `data/`: the input networks, all from the [SNAP](https://snap.stanford.edu/data/) network collection: Arxiv Astro, Arxiv Gr-Qc, Arxiv HepPh, Email-Enron, Facebook, Lastfm Asia.
- `results/`: the results shown in the paper.

## Setup

Tested with Python 3.13.7, networkx 3.5, numpy 2.2.6, scipy 1.16.1, pandas
2.3.2 and matplotlib 3.10.6.

```bash
make setup PYTHON=python3.13
```

Alternatively:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To use the notebook, also install Jupyter (`python -m pip install jupyter`).

## Running the experiments

Each script has its own `--help`. By default the scripts write to `outputs/`
(git-ignored), never to `results/`, so a re-run cannot overwrite the published
results.

```bash
source .venv/bin/activate

# NaDS vs CDS and baselines on one network (K = 5, 10, 15, 20; 200 s per unit of K)
python scripts/real_networks_comparison.py --network Facebook

# A quick smaller run
python scripts/real_networks_comparison.py --network Facebook --k-list 5 --time-per-k 5 --random-runs 2

# k-core and collective-influence baselines on every network
python scripts/kcore_ci_baselines.py

# Artificial LFR graphs
python scripts/lfr_experiments.py
```

The same runs are available as `make run-comparison NETWORK=Facebook`,
`make run-kcore-ci` and `make run-lfr`.

The solvers are given a wall-clock budget, so re-running them gives similar but
not bit-identical results. Random seeds are fixed (`numpy` 42, `random` 10).
The loaders convert the adjacency matrix to a dense array, so memory grows with
the square of the number of nodes: about 3 GB for Arxiv Astro and about 11 GB
for Email-Enron.

## Using the package

From the repository root:

```python
import sys
sys.path.insert(0, "src")

from nads import DEFAULT_PARAMS, NS_solver, SingleDiscount, create_graph

g, W, A = create_graph(size=[50, 50], probs=[[0.9, 0.1], [0.1, 0.9]], weights=0.1)
K = 3
_, x0 = SingleDiscount(g, W, DEFAULT_PARAMS, K)          # starting seed vector
s, X, history = NS_solver(g, W, x0, DEFAULT_PARAMS, delta=0.5, xi=0.1, d=2,
                          max_time=5, buffer_dim=500)
print(s[-1])
```

`params = (l0, h0, theta_l, theta_h, gamma, eps)` defines the propagation model;
`DEFAULT_PARAMS` holds the values used in the paper. A seed vector has `h0` on
the chosen nodes and `0` elsewhere.

## Results

`results/<network>/` contains:

- `ns_cds_disc/{ns,cds}_disc_<K>.csv`: solver history when started from the single-discount seeds, one row per improvement: `spread, elapsed seconds, influence evaluations`.
- `ns_cds_random/{ns,cds}_<K>.csv`: best spread found at each second (columns) for each of 10 random starting sets (rows).
- `scores.csv`, `times.csv`: baseline spread and running time, one row per method and one column per `K` in `5, 10, 15, 20`. The row order was not recorded in the original notebook.
- `greed.csv` (Arxiv Astro and HepPh only): greedy results.

The `- Copia.csv` files in `Arxiv Astro/` and `Arxiv HeP-Ph/` are additional variants of `scores.csv` and `times.csv` that were part of the original upload. They differ from the current files, and their provenance is not documented.

The scripts in `scripts/` write to `outputs/<network>/` with the same structure (plus a `baselines.csv` with the baseline scores and times).

## Citation

If you use this code or its results, please cite the paper: <https://doi.org/10.1093/comnet/cnaf042>.
