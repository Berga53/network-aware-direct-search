"""Baseline seed-selection methods the direct-search solvers are compared with.

Includes brute force, random search, greedy, single discount, RIS, and the
collective-influence (CI), k-core and Katz centrality heuristics.
"""

import itertools
import random
import time

import networkx as nx
import numpy as np

from .helpers import rand_bin_array
from .models import Influence_evaluation


def brute_force_solver(g, W, params, K):
  N = len(g.nodes)
  h0 = params[1]
  X = itertools.combinations(range(N), K)
  s = 0
  x_res = np.zeros(N)
  for elem in X:
    x_temp = np.zeros(N)
    x_temp[list(elem)]  = h0
    s_temp = Influence_evaluation(g, W, x_temp, params)[0]
    if s_temp > s:
      x_res = x_temp
      s = s_temp
  return s, x_res


def RS_solver(g, W, params, K, max_calls):
  N = len(g.nodes)
  calls = 0
  s = 0
  history = []
  start = time.time()

  while calls < max_calls:
    x_temp = rand_bin_array(K, N, params[1])
    s_temp =  Influence_evaluation(g, W, x_temp, params)[0]
    calls += 1
    print("\r" + "Random search... {}/{}. ETA: {} s.".format(calls, max_calls, round((time.time()-start)*((max_calls-calls)/calls))), end = "")

    if s_temp > s:
      s = s_temp
      X = x_temp
      history.append([s, calls])

  history.append([s, calls])

  print("\r" + "Random search... {}/{} Done!".format(calls, max_calls))
  print("s: {}".format(s))

  return s, X, history


def greedy(g, W, params, K):
  N = len(g.nodes)
  temp = [0.0 for i in range(N)]
  s_out = 0.0
  for i in range(K):
    idx = 0
    value = 0.0
    for k in range(N):
      if temp[k] == 0:
        temp2 = temp.copy()
        temp2[k] = params[1]
        s = Influence_evaluation(g, W, np.array(temp2), params)[0]
        if s > value:
          value = s
          idx = k
          s_out = s

    temp[idx] = params[1]

  return s_out, temp


def SingleDiscount(g,W, params, K):
  W1 = W.toarray()
  idx = []
  ddv = np.sum(W1 > 0, axis = 0)
  tv = np.zeros(len(g.nodes))
  for i in range(K):
    u = np.argmax(ddv)
    idx.append(u)
    v = list(np.nonzero(W1[u])[0])
    tv[v] += 1
    ddv[v] = ddv[v] - tv[v]
    ddv[u] = -1
    W1[u,:] = 0
    W1[:,u] = 0

  X = np.zeros(len(g.nodes))
  X[idx] = params[1]
  S = Influence_evaluation(g, W, X, params)[0]

  return S, X


def RIS(g,W, params, K, R):
  H = nx.Graph()
  H.add_nodes_from(g.nodes)
  for i in range(R):
    idx = random.randint(0, len(g.nodes)-1)
    temp = np.zeros(len(g.nodes))
    temp[idx] = params[1]
    x = Influence_evaluation(g, W, temp, params)[2][1:]
    edges = []
    for elem in x:
      edges += list(np.nonzero(elem)[0])
    edges = list(set(edges)-set([idx]))
    edges = [(idx, elem) for elem in edges]
    H.add_edges_from(edges)
  H = H.to_undirected()
  W1 = nx.adjacency_matrix(H).toarray()

  t = []
  for i in range(K):
    W_temp = np.sum(W1, axis = 0)
    v = np.argmax(W_temp)
    t.append(v)
    W1[v,:] = 0
    W1[:,v] = 0

  X = np.zeros(len(g.nodes))
  X[t] = params[1]
  S = Influence_evaluation(g, W, X, params)[0]

  return S, X


def collective_influence(G, d=3):
    assert d > 0
    ci = {}
    reduced_degrees = {node: G.degree(node) - 1 for node in G.nodes()}

    for node in G.nodes():
        ball = nx.single_source_shortest_path_length(G, node, cutoff=d)
        frontier = [n for n, dist in ball.items() if dist == d]
        ci_score = reduced_degrees[node] * sum(reduced_degrees.get(n, 0) for n in frontier)
        ci[node] = ci_score
    return ci

def approx_largest_eigenvalue(G, d, ci=None):
    if ci is None:
        ci = collective_influence(G, d)
    k_values = [G.degree(n) for n in G.nodes()]
    mean_ci = sum(ci.values()) / len(ci)
    mean_k = sum(k_values) / len(k_values)
    return (mean_ci / mean_k) ** (1 / (d + 1))

def get_influencers(G, d=3, max_influencers=None, verbose=False):
    assert nx.get_node_attributes(G, "name") == {} or len(G.nodes()) == len(set(G.nodes()))
    G = G.copy()
    influencers = []
    ci = collective_influence(G, d)
    ev = approx_largest_eigenvalue(G, d, ci)
    
    while ev > 1 and (max_influencers is None or len(influencers) < max_influencers):
        node_to_remove = max(ci, key=ci.get)
        influencers.append(node_to_remove)
        G.remove_node(node_to_remove)

        if verbose:
            print(f"Removed {node_to_remove}, CI={ci[node_to_remove]}, Eigenvalue={ev:.4f}")
        
        if len(G) == 0:
            break
        ci = collective_influence(G, d)
        ev = approx_largest_eigenvalue(G, d, ci)

    return influencers



def kcore_seeds(g, K, h0):
  """Seed vector with ``h0`` on the K nodes of highest core number."""
  x0 = np.zeros(len(g.nodes))
  x0[list(np.argsort(np.array(list(nx.core_number(g).values())))[-K:])] = h0
  return x0


def katz_seeds(g, K, h0):
  """Seed vector with ``h0`` on the K nodes of highest Katz centrality."""
  x0 = np.zeros(len(g.nodes))
  x0[list(np.argsort(np.array(list(nx.katz_centrality_numpy(g).values())))[-K:])] = h0
  return x0
