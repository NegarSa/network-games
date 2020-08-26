import numpy as np
import random
import networkx as nx
import copy
import matplotlib.pyplot as plt

class BestResponseNetworkGame():
    '''
        Implemets the base of a 2x2 network game
    '''

    def __init__(self, network, pi, init_state = None):
        '''
        Parameters
        ----------
        network : nx object
            The underlying network
        pi : 2d array
            The payoff matrices
        init_state : 1d array, optional
            Inital strategies.
        '''
        self.network = network
        self.pi = pi
        self.n = nx.number_of_nodes(self.network)
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
        self.x[i] = self.f(i)
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

    def nA(self, i):
        count = 0
        for v in nx.neighbors(self.network, i):
            if self.x[v] == 0: # if v is playing A
                count += 1
        return count

    def f(self, i):
        '''
        The update rule for agent i
        '''
        deg = self.net.degree(i)
        payoffs = self.pi[i]

        d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
        g = payoffs[3] - payoffs[1]

        if d == 0: return 0
        threshold = g / d
        nAi = self.nA(i)
        return int(threshold * deg >= nAi)

    def potential_i(self, i):
        '''
        The potential function for agent i

        Parameters
        ----------
        i : int,
            The agent i.
        '''
        payoffs = self.pi[i]
        d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
        g = payoffs[3] - payoffs[1]
        if d == 0: return 0
        threshold = g / d

        if self.x[i] == 0:
            return self.nA(i) - np.ceil(threshold * self.net.degree(i))
        else:
            return self.nA(i) - np.ceil(threshold * self.net.degree(i)) - 1

    def potential(self):
        '''
        Sum of potentials for all the agents, network potential.

        Parameters
        ----------
        None.
        '''
        r = 0
        for v in self.net.nodes():
            r += self.potential_i(v)
        return r

    def partition(self):
        '''
        Returns agent partitions.

        Parameters
        ----------
        None.
        '''
        sets_dict = {'AA': 0, 'AB': 0, 'BB': 0, 'BA': 0}

        for k in range(self.n):
            if self.x[k] == 0 and self.f_i(k) == 0: sets_dict['AA'] += 1
            elif self.x[k] == 0 and self.f_i(k) == 1: sets_dict['AB'] += 1
            elif self.x[k] == 1 and self.f_i(k) == 0: sets_dict['BA'] += 1
            elif self.x[k] == 1 and self.f_i(k) == 1: sets_dict['BB'] += 1

        return sets_dict

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
        self.x[i] = self.f(i)
        return self.x

    def targeted_control(self, method, p=4):
        '''
        Targeted control of network, until all agents play A.

        Parameters
        ----------
        method : string
            One of "IPRO", "distance", "EV"
        p : int, optional
            The power.
        '''

        total_incentives_given = 0

        for i in range(self.n):
            self.eq()

            max_ratio = 0
            max_agent = 0
            max_r = 0

            for k in range(self.n):
                if self.x[k] == 0: continue
                pi_copy = None
                payoffs = self.pi[k].copy()
                d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
                g = payoffs[3] - payoffs[1]

                if self.net.degree(k) == 0: r = g
                else: r = g - (((d * self.nA(k)) / self.net.degree(k)))

                if r <= 0: continue
                pi_copy = copy.deepcopy(self.pi)
                pi_copy[k][0] += r + 0.001
                pi_copy[k][1] += r + 0.001

                sim = None
                sim = br_network_game(self.net, pi_copy, self.x.copy())

                sim.eq()

                d_potential = sim.potential() - self.potential()

                if r == 0:
                    continue
                else:
                    ratio = d_potential / (r ** p)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_agent = k
                        max_r = r

            self.pi[max_agent][0] += max_r + 0.001
            self.pi[max_agent][1] += max_r + 0.001
            total_incentives_given += max_r

        mean = total_incentives_given/self.n
        #print(mean)
        return mean

    def targeted_dist(self, p=4):
        total_incentives_given = 0

        for i in range(self.n):
            self.eq()

            max_ev = 0
            max_agent = 0
            max_r = 0


            for k in range(self.n):
                if self.x[k] == 0: continue
                pi_copy = None
                payoffs = self.pi[k].copy()
                d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
                g = payoffs[3] - payoffs[1]

                if self.net.degree(k) == 0: r = g
                else: r = g - (((d * self.nA(k)) / self.net.degree(k)))

                if r <= 0: continue
                pi_copy = copy.deepcopy(self.pi)
                pi_copy[k][0] += r + 0.001
                pi_copy[k][1] += r + 0.001

                sim = None
                sim = br_network_game(self.net, pi_copy, self.x.copy())
                #one step:
                #sim.update(k)
                #until eq:
                sim.eq()

                state_self = list(self.partition().values())
                state_sim = list(sim.partition().values())

                dist = np.linalg.norm(np.array(state_self) - np.array(state_sim))

                if r == 0:
                    continue
                else:
                    ratio = dist / (r ** p)
                    if ratio > max_ev:
                        max_ev = ratio
                        max_agent = k
                        max_r = r

            self.pi[max_agent][0] += max_r + 0.001
            self.pi[max_agent][1] += max_r + 0.001
            total_incentives_given += max_r

#         print(total_incentives_given/self.n)
#         return self.x
        mean = total_incentives_given/self.n
        #print(mean)
        return mean

    def targeted_ev(self, p=4):
        total_incentives_given = 0

        for i in range(self.n):
            self.eq()

            #max_ev = 0
            min_agent = 0
            min_r = 0
            min_ev = 900

            for k in range(self.n):
                if self.x[k] == 0: continue
                pi_copy = None
                payoffs = self.pi[k].copy()
                d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
                g = payoffs[3] - payoffs[1]

                if self.net.degree(k) == 0: r = g
                else: r = g - (((d * self.nA(k)) / self.net.degree(k)))

                if r <= 0: continue
                pi_copy = copy.deepcopy(self.pi)
                pi_copy[k][0] += r + 0.001
                pi_copy[k][1] += r + 0.001

                sim = None
                sim = br_network_game(self.net, pi_copy, self.x.copy())
                #sim.update(k)
                sim.eq()

                s1 = list(self.partition().values())
                s2 = list(sim.partition().values())


                if self.x[k] == 0:
                    if s1[0] == 0: b = 0
                    else: b = (s1[0] - s2[0]) / s1[0]
                    if s1[3] == 0: c = 0
                    else: c = (s2[2] - s1[2] - 1) / s1[3]
                    if s1[1] == 0: a = 0
                    else: a = 1 / s1[1]
                    dist = min(1 - b, 1 - c, 1 - a)
                else:
                    if s1[1] == 0: b = 0
                    else: b = (s1[1] - s2[1]) / s1[1]
                    if s1[2] == 0: c = 0
                    else: c = (s1[2] - s2[2]) / s1[2]
                    if s1[3] == 0: a = 0
                    else: a = 1 / s1[3]
                    dist = min(1 - b, 1 - c, 1 - a)

                if r == 0 or dist == 1:
                    continue
                else:
                    ratio = dist / (r ** p)
                    if ratio < min_ev:
                        min_ev = ratio
                        min_agent = k
                        min_r = r

            self.pi[min_agent][0] += min_r + 0.001
            self.pi[min_agent][1] += min_r + 0.001
            total_incentives_given += min_r

#         print(total_incentives_given/self.n)
#         return self.x
        mean = total_incentives_given/self.n
        #print(mean)
        return mean

    def opt(self):

        agents_B = [v for v in self.net.nodes() if self.x[v] == 1]
        controlled_agents = []
        self.mostA = 0
        self.r_min = np.matrix(np.ones((self.n, 1)) * np.inf)

        # First recursive call
        self.next_agent(agents_B, controlled_agents, self.x.copy(),
                        list(np.zeros(self.n)), copy.deepcopy(self.pi))

        return sum(self.r_min) / self.n


    def next_agent(self, agents_B, controlled_agents, x, r, pi):

        for Bagent in agents_B:
            r_new = r
            pi_new = copy.deepcopy(pi)
            current_list = controlled_agents.copy()
            current_list.append(Bagent)
            payoffs = self.pi[Bagent].copy()
            d = payoffs[0] - payoffs[2] + payoffs[3] - payoffs[1]
            g = payoffs[3] - payoffs[1]

            if self.net.degree(Bagent) == 0: r_b = g
            else: r_b = g - (((d * self.nA(Bagent)) / self.net.degree(Bagent)))

            #r_b = self.r_i(Bagent)
            r_new[Bagent] = r_b + 0.0001
            pi_new[Bagent][0] += r_b + 0.0001
            pi_new[Bagent][1] += r_b + 0.0001

            new_game = br_network_game(self.net, pi_new, x)
            new_game.eq()
            x_new = new_game.x.copy()
            #print('x_new: ',x_new)
            #print(sum(x_new))
            if self.n - sum(x_new) >= self.mostA:
                if self.n - sum(x_new) > self.mostA or (self.n - sum(x_new) == self.mostA and sum(r_new) < sum(self.r_min)):
                    self.r_min = r_new.copy()
                self.mostA = self.n - sum(x_new)

            if sum(x_new) == 0: continue

            if len(agents_B) > 1:
                remB = agents_B.copy()
                remB.remove(Bagent)
                remB = [v for v in remB if x_new[v] == 1]

                #print(remB)
                #print(self.mostA)
                self.next_agent(remB, current_list, x_new.copy(), r_new.copy(), copy.deepcopy(pi_new))
