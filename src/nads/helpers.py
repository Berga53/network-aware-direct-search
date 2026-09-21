"""Graph construction, network loading and small shared utilities."""

import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import patches, pyplot

# Edge weight given to every edge of the real-world networks.
DEFAULT_ALPHA = 0.1

# Network name -> file name inside the data directory.
NETWORK_FILES = {
    "Arxiv_Astro": "Arxiv_Astro.txt",
    "Arxiv_GrQc": "Arxiv_GrQc.txt",
    "Arxiv_HepPh": "Arxiv_HepPh.txt",
    "Email-Enron": "Email-Enron.txt",
    "Facebook": "Facebook.txt",
    "Lastfm_asia": "Lastfm_asia.csv",
}


#function to drow the adjacency matrix
def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_array(G, dtype=bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)


def create_graph(size = [50,50], probs = [[0.9,0.1], [0.1,0.9]], weights = 0.1):
  temp = nx.stochastic_block_model(size, probs, seed=0)
  A = nx.adjacency_matrix(temp)
  W = A.multiply(weights).tocsr()
  G = nx.from_numpy_array(W)
  return G,W,A


def rand_bin_array(K, N, h):
  arr = np.zeros(N)
  arr[:K]  = h
  np.random.shuffle(arr)
  return arr


def history_to_np(history, max_calls):
  x = np.zeros(max_calls)
  t = 0
  temp = history[0][0]
  for i in range(max_calls):
    if history[t][1] == i+1:
      temp = history[t][0]
      if t != len(history)-1:
        t+=1
    x[i] = temp
  return x



def set_seeds(np_seed=42, py_seed=10):
  """Seed numpy and ``random`` with the values used for the paper."""
  np.random.seed(np_seed)
  random.seed(py_seed)


def load_network(name, data_dir):
  """Load one of the benchmark networks as a simple undirected graph."""
  path = Path(data_dir) / NETWORK_FILES[name]
  if path.suffix == ".csv":
    df = pd.read_csv(path, header=None, names=["source", "target"])
    g = nx.from_pandas_edgelist(df, source="source", target="target")
  else:
    g = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
  g = g.to_undirected()
  g.remove_edges_from(nx.selfloop_edges(g))
  return g


def weighted_graph(g, alpha=DEFAULT_ALPHA):
  """Give every edge weight ``alpha``; return the relabelled graph and its sparse weight matrix ``W``."""
  A = nx.adjacency_matrix(g)
  W = A.multiply(alpha)
  g = nx.from_numpy_array(W.toarray())
  return g, W


def best_so_far_curve(history, max_time):
  """Best influence found up to each second ``0..max_time`` from a solver ``history``."""
  t = []
  value = 0
  for j in range(max_time+1):
    for elem in history:
      if elem[1]> j:
        continue
      if elem[0] > value:
        value = elem[0]
    t.append(value)
  return t
