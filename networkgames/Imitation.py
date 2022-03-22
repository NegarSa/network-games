import copy

import networkx as nx
import numpy as np

from networkgames import NetworkGame


def generate_pi(n):
    r = []
    for i in range(n):
        w = []
        w = 1/2 * (np.random.uniform(0, 1, 4))
        w[0] += 1
        w[3] += 1
        r.append(list(w.copy()))
    return r


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


def f_im(network, x, pi, i):
    """
    The update rule for agent i (imitation)
    """

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


def potential(netgame):
    """
    Sum of potentials for all the agents, network potential.

    Parameters
    ----------
    None.
    """

    r = 0
    for v in netgame.network.nodes():
        r += netgame.nA(v)
    return r


def r_comp(netgame, i):
    max_j = 0
    max_t = 0
    i_neighbors = list(nx.neighbors(netgame.network, i))
    i_neighbors.extend([i])
    for j in i_neighbors:
        if netgame.x[j] == 1:
            max_t = 0
            j_neighbors = list(nx.neighbors(netgame.network, j))
            j_neighbors.extend([j])
            for t in j_neighbors:
                if netgame.x[t] == 1:
                    rh = total_payoff(netgame.network, netgame.x, netgame.pi, t) \
                         - total_payoff(netgame.network, netgame.x, netgame.pi, i)
                    print(rh)
                    if rh > max_t: max_t = rh
        if max_t > max_j: max_j = max_t
    return max_j


def A_any_B_playing_neighbor(netgame, i):
    """
        Checks if node i has the conditions to receive an incentive
        i.e. is playing A and has at least one neighbour playing B
    """

    if netgame.x[i] == 1: return False
    for v in nx.neighbors(netgame.network, i):
        if netgame.x[v] == 1:
            return True
    return False


def IPRO(netgame, p=4):
    """
            Targeted control of network, until all agents play A.

            Parameters
            ----------
            p : int, optional
                The power used in the IPRO algorithm.
            """

    if np.array_equal(netgame.x, np.ones(netgame.n, dtype=int)):
        # All agents playing 1, can't force anyone to switch
        return np.NaN

    total_incentives_given = 0

    while not np.array_equal(netgame.x, np.zeros(netgame.n, dtype=int)):
        netgame.eq()

        max_ratio = 0
        max_agent = 0
        max_r = 0

        for k in range(netgame.n):
            netgame.eq()
            if netgame.x[k] == 0: continue
            if A_any_B_playing_neighbor(netgame, k): continue # can we switch this agent?

            r = r_comp(netgame, k)

            if r <= 0: continue
            pi_copy = None
            pi_copy = copy.deepcopy(netgame.pi)
            pi_copy[k][0] += r + 0.001
            pi_copy[k][1] += r + 0.001

            sim = None
            sim = NetworkGame.NetworkGame(netgame.network, pi_copy, f_im, netgame.x.copy())

            sim.eq()

            d_potential = potential(sim) - potential(netgame)
            #print(d_potential)

            ratio = d_potential / (r ** p)
            if ratio > max_ratio:
                max_ratio = ratio
                max_agent = k
                max_r = r

        netgame.pi[max_agent][0] += max_r + 0.001
        netgame.pi[max_agent][1] += max_r + 0.001
        total_incentives_given += max_r

    mean = total_incentives_given / netgame.n
    #print(mean)
    return mean


def Distance(netgame, p=4):
    """
            Targeted control of network, until all agents play A.

            Parameters
            ----------
            p : int, optional
                The power used in the IPRO algorithm.
            """

    if np.array_equal(netgame.x, np.ones(netgame.n, dtype=int)):
        # All agents playing 1, can't force anyone to switch
        return np.NaN

    total_incentives_given = 0

    while not np.array_equal(netgame.x, np.zeros(netgame.n, dtype=int)):
        netgame.eq()

        max_ratio = 0
        max_agent = 0
        max_r = 0

        for k in range(netgame.n):
            netgame.eq()
            if netgame.x[k] == 0: continue
            if A_any_B_playing_neighbor(netgame, k): continue # can we switch this agent?

            r = r_comp(netgame, k)

            if r <= 0: continue
            pi_copy = None
            pi_copy = copy.deepcopy(netgame.pi)
            pi_copy[k][0] += r + 0.001
            pi_copy[k][1] += r + 0.001

            sim = None
            sim = NetworkGame.NetworkGame(netgame.network, pi_copy, f_im, netgame.x.copy())

            sim.eq()

            state_self = list(netgame.partition().values())
            state_sim = list(sim.partition().values())
            distance = np.abs((state_self) - (state_sim))

            ratio = distance / (r ** p)
            if ratio > max_ratio:
                max_ratio = ratio
                max_agent = k
                max_r = r

        netgame.pi[max_agent][0] += max_r + 0.001
        netgame.pi[max_agent][1] += max_r + 0.001
        total_incentives_given += max_r

    mean = total_incentives_given / netgame.n
    #print(mean)
    return mean
