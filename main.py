from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def drawGraph():
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


# init graph
G = nx.DiGraph()

buf = []
assign = []

size = int(input("how many nodes:"))
buf = [[0 for i in range(size)] for j in range(size)]
assign = [[False for i in range(size)] for j in range(size)]


def getInput():
    for i in range(size):
        for j in range(size):
            # ignore the diagonal
            if i == j:
                continue
            # ignore the weight if it's already set
            # 1-2 = 2-1
            if assign[i][j] == True or assign[j][i] == True:
                continue
            buf[i][j] = int(input(f"Enter the weight of {i+1} to {j+1}: "))
            assign[i][j] = True


# getInput()

# 5x5 testdata
# buf = [
#     [0, 9, 8, 10, 0],
#     [0, 0, 12, 13, 7],
#     [0, 0, 0, 20, 0],
#     [0, 0, 0, 0, 2],
#     [0, 0, 0, 0, 0],
# ]


# 7x7 testdata
buf = [
    [0, 20, 0, 0, 2, 8, 12],
    [0, 0, 0, 0, 2, 8, 12],
    [0, 0, 0, 0, 3, 10, 18],
    [0, 0, 0, 0, 3, 10, 18],
    [0, 0, 0, 0, 0, 8, 7],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]

# show the matrix
print(buf)

if len(buf) < 2:
    print("Not enough nodes")
    exit()


def getLength(buf=[], i=0, j=0):
    v = buf[i][j]
    if v == 0:
        v = buf[j][i]
    return v


### Step 1
highest = 0
nodes = []
for i in range(size):
    for j in range(size):
        if buf[i][j] > highest:
            highest = buf[i][j]
            nodes = [i, j]

print(f"Max value is {highest} {nodes[0]+1} to {nodes[1]+1}")
G.add_edges_from(
    [
        (
            nodes[0] + 1,
            nodes[1] + 1,
        ),
    ],
    weight=highest,
)


# Step 2 recursive

steps = []


def findChildren(toggle=False):
    if len(nodes) >= size - 1:
        return

    available = 0
    new_list = nodes[:]

    comp = list(combinations(new_list, 3))

    if len(nodes) == 2:
        comp = [nodes]
    print("comp=>", comp, "nodes=>", nodes)

    for c in comp:
        tmp = 0
        # z is not in nodes
        z = 0
        for i in range(size):
            s = 0
            if i in c:
                continue
            for j in nodes:
                w = getLength(buf, i, j)
                s += w
            if s > tmp:
                tmp = s
                z = i
        available = z

    for n in nodes:
        w = getLength(buf, n, available)
        # if w == 0:
        #     continue
        # check if start already connect to
        nn = n + 1
        tt = available + 1
        start = nn
        end = tt
        if toggle:
            start = nn
            end = tt

        G.add_edges_from(
            [
                (
                    start,
                    end,
                ),
            ],
            weight=w,
        )
        if not nx.is_planar(G):
            G.remove_edge(start, end)
            continue
        else:
            print("connect ", start, " to ", end, " with weight ", w)

    # add z to nodes
    nodes.append(available)
    steps.append(available + 1)


t = False

# o = 1
# # Step 3
# while len(nodes) < size - 1:
#     if o == 6:
#         break
#     findChildren(t)
#     t = not t

#     o = o + 1


findChildren(False)
findChildren(True)

# Final step


pos = nx.planar_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    font_weight="bold",
    node_color="lightblue",
)

plt.show()
