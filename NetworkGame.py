import numpy as np
import random
import networkx as nx
import copy
import matplotlib.pyplot as plt

class NetworkGame():
    '''
        Implemets the base of a 2x2 network game
    '''

    def __init__(self, network, pi, f, init_state = None):
        '''
        Parameters
        ----------
        network : nx object
            The underlying network
        pi : 2d array
            The payoff matrices
        f : fucntion
            The update rule
        init_state : 1d array, optional
            Inital strategies.
        '''
        self.network = network
        self.pi = pi
        self.n = nx.number_of_nodes(self.network)
        self.f = f
        if init_state is None:
            self.x = np.ones(self.n)
        else:
            self.x = init_state.copy()

    def update(self, i = None):
        '''
        Updating an agent.

        Parameters
        ----------
        i : int, optional
            The agent. If none is given, a random agent is updated.
        '''
        if i is None:
            i = random.randint(0, self.n - 1)
        self.x[i] = self.f(self.x, self.pi, i)
        return self.x

    def eq(self):
        '''
        Updates agents until reaching an equilibrium.
        (Updates each agent a number of times to make sure an eq. is reached.)

        Parameters
        ----------
        None.
        '''
        random_seq_start = np.random.permutation(self.n)
        for j in range(self.n * 3):
            self.update(random_seq_start[j % self.n])

    def draw(self):
        '''
        Draws graph of the network, colored with the agents' strategies.

        Parameters
        ----------
        None.
        '''
        nx.draw(self.net, node_size=200, node_color=self.x + 0.5, with_labels=True)

    def partition(self):
        '''
        Returns agent partitions.

        Parameters
        ----------
        None.
        '''
        sets_dict = {'AA': 0, 'AB': 0, 'BB': 0, 'BA': 0}

        for k in range(self.n):
            if self.x[k] == 0 and self.f(self.x, self.pi, i) == 0:
                sets_dict['AA'] += 1
            elif self.x[k] == 0 and self.f(self.x, self.pi, i) == 1:
                sets_dict['AB'] += 1
            elif self.x[k] == 1 and self.f(self.x, self.pi, i) == 0:
                sets_dict['BA'] += 1
            elif self.x[k] == 1 and self.f(self.x, self.pi, i) == 1:
                sets_dict['BB'] += 1

        return sets_dict
