"""Direct-search solvers for influence maximisation.

``NS_solver`` is NaDS, the network-aware direct search: it first explores the
neighbours of the seeds in the graph, and falls back to the generic swap
neighbourhood (as in ``CDS_solver``, the custom direct search) when none
improves. Throughout the code and results ``NS`` stands for NaDS.

Each solver comes in two variants that differ only in the budget:
``*_solver`` stops after ``max_time`` seconds (used for the paper's
experiments) and ``*_solver_calls`` after ``max_calls`` influence evaluations.
The two ``history`` formats differ accordingly: rows are
``[spread, elapsed_seconds, calls]`` for the time-budget variants and
``[spread, calls]`` for the call-budget ones.
"""

import collections
import itertools
import time

import numpy as np

from .models import Influence_evaluation


def CDS_solver_calls(g, W, x0, params, delta, xi, d, max_calls, buffer_dim):

  N = len(x0)
  K = np.count_nonzero(x0)
  X = [x0]
  s = [Influence_evaluation(g, W, x0, params)[0]]
  r = 0
  xi_t = xi
  l0, h0, theta_l, theta_h, gamma, eps = params

  stop = False
  buffer = collections.deque(maxlen = buffer_dim)
  calls = 1
  start = time.time()
  history = [[s[-1],calls]]

  print("\r" + "Custom direct search... {}/{}. ETA: {} s.".format(calls, max_calls, round((time.time()-start)*((max_calls-calls)/calls))), end = "")

  while (stop == False) and (r<1000):
    idx = set(np.nonzero(X[-1])[0])
    neighbors = []
    idx_temp = set(range(N))-set(idx)

    for i in range(d//2):
      for elem1 in itertools.combinations(idx, len(idx)-1-i):
        for elem2 in itertools.combinations(idx_temp, i+1):
          neighbors.append(list(set(elem1) | set(elem2)))


    s_temp = s[-1]
    x_temp = X[-1]

    for elem in neighbors:
      x_elem = np.zeros(N)
      x_elem[list(elem)]  = h0
      if list(x_elem) in buffer:
        continue
      if calls >= max_calls:
        stop = True
        break

      s_elem = Influence_evaluation(g, W, x_elem, params)[0]
      buffer.append(list(x_elem))
      calls += 1
      print("\r" + "Custom direct search... {}/{}. ETA: {} s.".format(calls, max_calls, round((time.time()-start)*((max_calls-calls)/calls))), end = "")

      if s_elem > s_temp:
        history.append([s_elem,calls])
        x_temp = x_elem
        s_temp = s_elem
        if s_temp > (1+xi_t)*s[-1]:
          break

    X.append(x_temp)
    s.append(s_temp)
    r += 1

    if s[-1] == s[-2]:
      X.pop()
      s.pop()
      stop = True
    elif s[-1] > (1+xi_t)*s[-2]:
      continue
    else:
      xi_t = xi_t * delta

  history.append([s[-1], calls])

  print("\r" + "Custom direct search... {}/{} Done!".format(calls, max_calls))
  print("s: {}".format(s[-1]))

  return s, X, history


def CDS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim):

  N = len(x0)
  K = np.count_nonzero(x0)
  X = [x0]
  s = [Influence_evaluation(g, W, x0, params)[0]]
  r = 0
  xi_t = xi
  l0, h0, theta_l, theta_h, gamma, eps = params

  stop = False
  buffer = collections.deque(maxlen = buffer_dim)
  calls = 1
  start = time.time()
  history = [[s[-1],time.time()-start, calls]]

  print("\r" + "Custom direct search... Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]), end = "")

  while (stop == False) and (r<1000):
    idx = set(np.nonzero(X[-1])[0])
    neighbors = []
    idx_temp = set(range(N))-set(idx)

    for i in range(d//2):
      for elem1 in itertools.combinations(idx, len(idx)-1-i):
        for elem2 in itertools.combinations(idx_temp, i+1):
          neighbors.append(list(set(elem1) | set(elem2)))


    s_temp = s[-1]
    x_temp = X[-1]

    for elem in neighbors:
      x_elem = np.zeros(N)
      x_elem[list(elem)]  = h0
      if list(x_elem) in buffer:
        continue
      if time.time()-start >= max_time:
        stop = True
        break

      s_elem = Influence_evaluation(g, W, x_elem, params)[0]
      buffer.append(list(x_elem))
      calls += 1
      print("\r" + "Custom direct search... Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]), end = "")

      if s_elem > s_temp:
        history.append([s_elem,time.time()-start, calls])
        x_temp = x_elem
        s_temp = s_elem
        if s_temp > (1+xi_t)*s[-1]:
          break

    X.append(x_temp)
    s.append(s_temp)
    r += 1

    if s[-1] == s[-2]:
      X.pop()
      s.pop()
      stop = True
    elif s[-1] > (1+xi_t)*s[-2]:
      continue
    else:
      xi_t = xi_t * delta

  history.append([s[-1], time.time()-start, calls])   

  print("\r" + "Custom direct search... Done! Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]))

  return s, X, history


def NS_solver_calls(g, W, x0, params, delta, xi, d, max_calls, buffer_dim):

  N = len(x0)
  K = np.count_nonzero(x0)
  X = [x0]
  s = [Influence_evaluation(g, W, x0, params)[0]]
  r = 0
  xi_t = xi
  l0, h0, theta_l, theta_h, gamma, eps = params

  stop = False
  buffer = collections.deque(maxlen = buffer_dim)
  calls = 1
  start = time.time()
  history = [[s[-1],calls]]

  print("\r" + "Neighbors search... {}/{}. ETA: {} s.".format(calls, max_calls, round((time.time()-start)*((max_calls-calls)/calls))), end = "")

  while (stop == False) and (r<1000):
    idx = set(np.nonzero(X[-1])[0])
    neighbors = []

    for elem1 in idx:
      for elem2 in set(g.neighbors(elem1))-idx:
        temp = idx.copy()
        temp.add(elem2)
        temp.remove(elem1)
        neighbors.append(temp)

    s_temp = s[-1]
    x_temp = X[-1]

    for elem in neighbors:
      x_elem = np.zeros(N)
      x_elem[list(elem)]  = h0
      if list(x_elem) in buffer:
        continue
      if calls >= max_calls:
        stop = True
        break

      s_elem = Influence_evaluation(g, W, x_elem, params)[0]
      buffer.append(list(x_elem))
      calls +=1
      print("\r" + "Neighbors search... {}/{}. ETA: {} s.".format(calls, max_calls, round((time.time()-start)*((max_calls-calls)/calls))), end = "")

      if s_elem > s_temp:
        history.append([s_elem,calls])
        x_temp = x_elem
        s_temp = s_elem
        if s_temp > (1+xi_t)*s[-1]:
          break

    X.append(x_temp)
    s.append(s_temp)
    r += 1

    if s[-1] > (1+xi_t)*s[-2]:
      continue

    elif s[-1] > s[-2]:
      xi_t = xi_t * delta

    else:
      X.pop()
      s.pop()
      idx = set(np.nonzero(X[-1])[0])
      neighbors = []
      idx_temp = set(range(N))-set(idx)

      for i in range(d//2):
        for elem1 in itertools.combinations(idx, len(idx)-1-i):
          for elem2 in itertools.combinations(idx_temp, i+1):
            neighbors.append(list(set(elem1) | set(elem2)))

      s_temp = s[-1]
      x_temp = X[-1]

      for elem in neighbors:
        x_elem = np.zeros(N)
        x_elem[list(elem)]  = h0
        if list(x_elem) in buffer:
          continue
        if calls >= max_calls:
          stop = True
          break

        s_elem = Influence_evaluation(g, W, x_elem, params)[0]
        buffer.append(list(x_elem))
        calls += 1
        print("\r" + "Neighbors search... {}/{}. ETA: {} s.".format(calls, max_calls, round((time.time()-start)*((max_calls-calls)/calls))), end = "")

        if s_elem > s_temp:
          history.append([s_elem,calls])
          x_temp = x_elem
          s_temp = s_elem
          if s_temp > (1+xi_t)*s[-1]:
            break

      X.append(x_temp)
      s.append(s_temp)
      r += 1

      if s[-1] == s[-2]:
        X.pop()
        s.pop()
        stop = True

      elif s[-1] > (1+xi_t)*s[-2]:
        continue

      else:
        xi_t = xi_t * delta

  history.append([s[-1], calls])

  print("\r" + "Neighbors search... {}/{} Done!".format(calls, max_calls))
  print("s: {}".format(s[-1]))

  return s, X, history


def NS_solver(g, W, x0, params, delta, xi, d, max_time, buffer_dim):

  N = len(x0)
  K = np.count_nonzero(x0)
  X = [x0]
  s = [Influence_evaluation(g, W, x0, params)[0]]
  r = 0
  xi_t = xi
  l0, h0, theta_l, theta_h, gamma, eps = params

  stop = False
  buffer = collections.deque(maxlen = buffer_dim)
  calls = 1
  start = time.time()
  history = [[s[-1],time.time()-start, calls]]

  print("\r" + "Neighbors search... Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]), end = "")

  while (stop == False) and (r<1000):
    idx = set(np.nonzero(X[-1])[0])
    neighbors = []

    for elem1 in idx:
      for elem2 in set(g.neighbors(elem1))-idx:
        temp = idx.copy()
        temp.add(elem2)
        temp.remove(elem1)
        neighbors.append(temp)

    s_temp = s[-1]
    x_temp = X[-1]

    for elem in neighbors:
      x_elem = np.zeros(N)
      x_elem[list(elem)]  = h0
      if list(x_elem) in buffer:
        continue
      if time.time()-start >= max_time:
        stop = True
        break

      s_elem = Influence_evaluation(g, W, x_elem, params)[0]
      buffer.append(list(x_elem))
      calls +=1
      print("\r" + "Neighbors search... Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]), end = "")

      if s_elem > s_temp:
        history.append([s_elem, time.time()-start, calls])
        x_temp = x_elem
        s_temp = s_elem
        if s_temp > (1+xi_t)*s[-1]:
          break

    X.append(x_temp)
    s.append(s_temp)
    r += 1

    if s[-1] > (1+xi_t)*s[-2]:
      continue

    elif s[-1] > s[-2]:
      xi_t = xi_t * delta

    else:
      X.pop()
      s.pop()
      idx = set(np.nonzero(X[-1])[0])
      neighbors = []
      idx_temp = set(range(N))-set(idx)

      for i in range(d//2):
        for elem1 in itertools.combinations(idx, len(idx)-1-i):
          for elem2 in itertools.combinations(idx_temp, i+1):
            neighbors.append(list(set(elem1) | set(elem2)))

      s_temp = s[-1]
      x_temp = X[-1]

      for elem in neighbors:
        x_elem = np.zeros(N)
        x_elem[list(elem)]  = h0
        if list(x_elem) in buffer:
          continue
        if time.time()-start >= max_time:
          stop = True
          break

        s_elem = Influence_evaluation(g, W, x_elem, params)[0]
        buffer.append(list(x_elem))
        calls += 1
        print("\r" + "Neighbors search... Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]), end = "")
        if s_elem > s_temp:
          history.append([s_elem, time.time() - start, calls])
          x_temp = x_elem
          s_temp = s_elem
          if s_temp > (1+xi_t)*s[-1]:
            break

      X.append(x_temp)
      s.append(s_temp)
      r += 1

      if s[-1] == s[-2]:
        X.pop()
        s.pop()
        stop = True

      elif s[-1] > (1+xi_t)*s[-2]:
        continue

      else:
        xi_t = xi_t * delta

  history.append([s[-1], time.time()-start, calls])

  print("\r" + "Neighbors search... Done! Calls:{}. Time: {}/{} s. Influence spread: {}.".format(calls, round(time.time()-start), max_time, s[-1]))

  return s, X, history
