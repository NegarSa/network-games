import random

import networkx as nx
import numpy as np


def update_rule(n):
    v = random.choice(n, [0, 1], p=1/2)
    return v

def total_payoff(network, x, pi, i):
    nAi = 0
    for v in nx.neighbors(network, i):
        if x[v] == 0:  # if v is playing A
            nAi += 1
    nBi = len(list(nx.neighbors(network, i))) - nAi
    if x[i] == 0:
        return (pi[i][0] * nAi) + (pi[i][1] * nBi)
    else:
        return (pi[i][2] * nAi) + (pi[i][3] * nBi)

def generate_pi(n, v):
    r = []
    for i in range(n):
        if v[i] == 0:
            p = []
            t_i = random.uniform(0, 2 / 3)
            p.append((1 - t_i) / t_i)
            p.append(0)
            p.append(0)
            p.append(1)
            r.append(p)
        else:
            w = []
            w = 1 / 2 * (np.random.uniform(0, 1, 4))
            w[0] += 1
            w[3] += 1
            r.append(list(w.copy()))
    return r


def f_br(network, x, pi, i, v):
    """
    The update rule for agent i
    """

    def nA():
        count = 0
        for v in nx.neighbors(network, i):
            if x[v] == 0:  # if v is playing A
                count += 1
        return count

    if v[i] == 0:
        deg = network.degree(i)
        payoffs = pi[i]

        d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
        g = payoffs[3] - payoffs[1]

        if d == 0: return 0
        threshold = g / d
        nAi = nA()
        return int(threshold * deg >= nAi)
    else:
        ni = list(nx.neighbors(network, i))
        ni.extend([i])
        utilities = [(k, total_payoff(network, x, pi, k)) for k in ni]
        maxu = 0
        imit = 0
        for e, u in utilities:
            if u > maxu:
                maxu = u
                imit = e

        return x[imit]


