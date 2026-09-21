"""Influence propagation model and the paper's default parameters.

``params`` is the tuple ``(l0, h0, theta_l, theta_h, gamma, eps)`` used
throughout the package: ``l0``/``h0`` are the base lower/upper thresholds of
the activation function (``h0`` is also the value given to every seed),
``theta_l``/``theta_h`` scale them at each round, ``gamma`` enters the
stopping test as a per-round discount and ``eps`` is the stopping tolerance.
"""

import numpy as np

# Values used for every experiment in the paper.
DEFAULT_PARAMS = (1, 1, 2, 50, 0, 0.1)


def GIP_function(x,h,l):
  r = np.multiply(x >= l, x)
  return np.minimum(r, h)


def Influence_evaluation2(g, W, x0, params, max_t = 999):

  l0, h0, theta_l, theta_h, gamma, eps = params

  s = 0

  st = [0]

  x = [x0]

  alpha = 0.1

#   for u, v, d in g.edges(data=True):
#     alpha += d["weight"]

#   alpha = alpha/len(g.edges)

  t = 1

  while np.linalg.norm(x[-1]*((1-gamma)**t)) > eps and t <= max_t:

    l = ((theta_l*alpha)**t)*l0
    h = (theta_h*(theta_l**(t-1))*(alpha**t))*h0

    xt = GIP_function(np.sum(np.multiply(W,x[-1]), axis = 1), h, l)
    s += np.sum(xt)

    x.append(xt)

    st.append(s)
    t += 1

  return s, st, x


def Influence_evaluation(g, W, x0, params, max_t = 999):

  l0, h0, theta_l, theta_h, gamma, eps = params


  s = 0

  st = [0]

  x = [x0]

  alpha = 0.1

  # for u, v, d in g.edges(data=True):
  #   alpha += d["weight"]

  # alpha = alpha/len(g.edges)

  t = 1


  while np.linalg.norm(x[-1]*((1-gamma)**t)) > eps and t <= max_t:


    l = ((theta_l*alpha)**t)*l0
    h = (theta_h*(theta_l**(t-1))*(alpha**t))*h0

    xt = GIP_function(W.multiply(x[-1]).sum(axis = 1), h, l)
    s += np.sum(xt)

    x.append(xt)

    st.append(s)
    t += 1

  return s, st, x
