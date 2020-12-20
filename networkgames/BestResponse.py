import copy
import random

import networkx as nx
import numpy as np

from networkgames import NetworkGame


def generate_pi(n):
    r = []
    for i in range(n):
        p = []
        t_i = random.uniform(0, 2 / 3)
        p.append((1 - t_i) / t_i)
        p.append(0)
        p.append(0)
        p.append(1)
        r.append(p)

    return r


def f_br(network, x, pi, i):
    """
    The update rule for agent i
    """

    def nA():
        count = 0
        for v in nx.neighbors(network, i):
            if x[v] == 0:  # if v is playing A
                count += 1
        return count

    deg = network.degree(i)
    payoffs = pi[i]

    d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
    g = payoffs[3] - payoffs[1]

    if d == 0: return 0
    threshold = g / d
    nAi = nA()
    return int(threshold * deg >= nAi)


def potential_i(netgame, i):
    """
    The potential function for agent i

    Parameters
    ----------
    i : int,
        The agent i.
    """
    payoffs = netgame.pi[i]
    d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
    g = payoffs[3] - payoffs[1]
    if d == 0: return 0
    threshold = g / d

    if netgame.x[i] == 0:
        return netgame.nA(i) - np.ceil(threshold * netgame.network.degree(i))
    else:
        return netgame.nA(i) - np.ceil(threshold * netgame.network.degree(i)) - 1


def potential(netgame):
    """
    Sum of potentials for all the agents, network potential.

    Parameters
    ----------
    None.
    """
    r = 0
    for v in netgame.network.nodes():
        r += potential_i(netgame, v)
    return r


def IPRO(netgame, p=4):
    """
            Targeted control of network, until all agents play A.

            Parameters
            ----------
            p : int, optional
                The power used in the IPRO algorithm.
            """

    total_incentives_given = 0

    for i in range(netgame.n):
        netgame.eq()

        max_ratio = 0
        max_agent = 0
        max_r = 0

        for k in range(netgame.n):
            if netgame.x[k] == 0: continue
            pi_copy = None
            payoffs = netgame.pi[k].copy()
            d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
            g = payoffs[3] - payoffs[1]

            if netgame.network.degree(k) == 0: r = g
            else: r = g - (((d * netgame.nA(k)) / netgame.network.degree(k)))

            if r <= 0: continue
            pi_copy = copy.deepcopy(netgame.pi)
            pi_copy[k][0] += r + 0.001
            pi_copy[k][1] += r + 0.001

            sim = None
            sim = NetworkGame.NetworkGame(netgame.network, pi_copy, f_br, netgame.x.copy())

            sim.eq()

            d_potential = potential(sim) - potential(netgame)
            # print(d_potential)

            if r == 0:
                continue
            else:
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
                The power used in the Distance algorithm.
            """

    total_incentives_given = 0

    for i in range(netgame.n):
        netgame.eq()

        max_ratio = 0
        max_agent = 0
        max_r = 0

        for k in range(netgame.n):
            if netgame.x[k] == 0: continue
            pi_copy = None
            payoffs = netgame.pi[k].copy()
            d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
            g = payoffs[3] - payoffs[1]

            if netgame.network.degree(k) == 0: r = g
            else: r = g - (((d * netgame.nA(k)) / netgame.network.degree(k)))

            if r <= 0: continue
            pi_copy = copy.deepcopy(netgame.pi)
            pi_copy[k][0] += r + 0.001
            pi_copy[k][1] += r + 0.001

            sim = None
            sim = NetworkGame.NetworkGame(netgame.network, pi_copy, f_br, netgame.x.copy())

            sim.eq()

            state_self = list(netgame.partition().values())
            state_sim = list(sim.partition().values())
            distance = np.linalg.norm(np.array(state_self) - np.array(state_sim))

            if r == 0:
                continue
            else:
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