import networkx as nx
import numpy as np

from networkgames import NetworkGame, BestResponse

n = 100

net = nx.generators.random_geometric_graph(n, np.sqrt(11. / (np.pi * 10)))
pi = BestResponse.generate_pi(n)
br_game = NetworkGame.NetworkGame(net, pi, BestResponse.f_br)
print(br_game.x)
print(BestResponse.IPRO(br_game))
print(br_game.x)

