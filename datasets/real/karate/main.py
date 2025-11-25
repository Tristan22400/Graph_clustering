import networkx as nx

nxg = nx.karate_club_graph()
nx.write_gml(nxg, "karate.gml")
