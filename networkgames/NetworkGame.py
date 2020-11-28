import random

import networkx as nx
import numpy as np


class NetworkGame:
    """
        Implements the base of a 2x2 network game
    """

    def __init__(self, network, pi, update_rule, init_state=None):
        """
        Parameters
        ----------
        network : nx object
            The underlying network
        pi : 2d array
            The payoff matrices
        update_rule : function
            The update rule (one function for each agent)
        init_state : 1d array, optional
            Initial strategies.
        """
        self.network = network
        self.pi = pi
        self.n = nx.number_of_nodes(self.network)
        self.update_rule = update_rule
        if init_state is None:
            self.x = np.random.choice([0, 1], size=(self.n,), p=[1. / 2, 1. / 2])
        else:
            self.x = init_state.copy()

    def f(self, i):
        return self.update_rule(self.network, self.x, self.pi, i)

    def update(self, i=None):
        """
        Updating an agent.

        Parameters
        ----------
        i : int, optional
            The agent. If none is given, a random agent is updated.
        """
        if i is None:
            i = random.randint(0, self.n - 1)
        self.x[i] = self.f(i)
        return self.x

    def eq(self):
        """
        Updates agents until reaching an equilibrium.
        (Updates each agent a number of times to make sure an eq. is reached.)

        Parameters
        ----------
        None.
        """
        random_seq_start = np.random.permutation(self.n)
        for j in range(self.n * 3):
            self.update(random_seq_start[j % self.n])

    def draw(self):
        """
        Draws graph of the network, colored with the agents' strategies.

        Parameters
        ----------
        None.
        """
        nx.draw(self.network, node_size=200, node_color=self.x + 0.5, with_labels=True)

    def partition(self):
        """
        Returns agent partitions.

        Parameters
        ----------
        None.
        """
        sets_dict = {'AA': 0, 'AB': 0, 'BB': 0, 'BA': 0}

        for k in range(self.n):
            if self.x[k] == 0 and self.f(k) == 0:
                sets_dict['AA'] += 1
            elif self.x[k] == 0 and self.f(k) == 1:
                sets_dict['AB'] += 1
            elif self.x[k] == 1 and self.f(k) == 0:
                sets_dict['BA'] += 1
            elif self.x[k] == 1 and self.f(k) == 1:
                sets_dict['BB'] += 1

        return sets_dict

    def nA(self, i):
        count = 0
        for v in nx.neighbors(self.network, i):
            if self.x[v] == 0:  # if v is playing A
                count += 1
        return count
