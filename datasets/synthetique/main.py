import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
n = 250
tau1 = 3
tau2 = 1.5
mu = 0.01
for mu in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    nxg = LFR_benchmark_graph(
        n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10
    )
    communities = list({frozenset(nxg.nodes[v]["community"]) for v in nxg})
    for node in range(nxg.number_of_nodes()):
        for comidx, com in enumerate(communities):
            if nxg.nodes[node]["community"] == com:
                nxg.nodes[node]["value"] = comidx
        nxg.nodes[node].pop("community", None)
    nx.write_gml(nxg, f"lfr_{mu:.2f}.gml")