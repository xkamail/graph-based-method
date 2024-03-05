import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

G.add_edges_from(
    [
        (
            3,
            4,
        ),
    ],
    weight=20,
)

G.add_edges_from(
    [
        (
            3,
            2,
        ),
    ],
    weight=12,
)

G.add_edges_from(
    [
        (
            2,
            4,
        ),
    ],
    weight=13,
)

G.add_edges_from([(2, 1)], weight=9)
G.add_edges_from([(3, 1)], weight=8)
G.add_edges_from([(4, 1)], weight=10)


# add 5 betwen 2,1,4
G.add_edges_from([(5, 2)], weight=7)
G.add_edges_from([(5, 1)], weight=0)
G.add_edges_from([(5, 4)], weight=2)


pos = nx.planar_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    font_weight="bold",
    node_color="lightblue",
)

plt.show()
