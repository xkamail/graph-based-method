from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# init graph

buf = []
assign = []

# size = int(input("how many nodes:"))
size = 7
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


G = nx.Graph()  # graph for calculation not render
Gx = nx.DiGraph()  # graph for render

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
brunch = [(nodes[0] + 1, nodes[1] + 1)]


def joinNode(node_index, wings=[0, 0, 0]):
    if len(wings) == 2:
        # (3, 2),
        # (2, 4),
        brunch.append((wings[0] + 1, node_index + 1))
        brunch.append((node_index + 1, wings[1] + 1))

    for w in sorted(wings):
        G.add_edges_from(
            [
                (
                    w + 1,
                    node_index + 1,
                ),
            ],
            weight=getLength(buf, node_index, w),
        )
        if len(wings) == 3:
            start, end = 0, 0
            if len(nodes) % 2 != 0:
                start = w + 1
                end = node_index + 1
            else:
                start = node_index + 1
                end = w + 1
            brunch.append((start, end))

    nodes.append(node_index)
    print("Current Nodes:", [node + 1 for node in nodes])
    return


def calculatePivots():

    if len(nodes) <= 2:
        # create size-2 x 2 matrix
        print(f"create size-2 x 2 matrix")
        return [[nodes[0], nodes[1]]]

    if len(nodes) == 3:
        return [nodes]
    print("we have more than 3 nodes", [x + 1 for x in nodes])
    cycles = []
    for x in list(nx.enumerate_all_cliques(G)):
        if len(x) != 3:
            continue
        # if cycle contains nodes[0], nodes[1], nodes[2]
        if (nodes[0] + 1) in x and (nodes[1] + 1) in x and (nodes[2] + 1) in x:
            continue
        cycles.append([node - 1 for node in x])
    return cycles[:]


# Step 2 recursive


def findNext():
    print(f"\n\n\n")
    pivots = calculatePivots()

    large_of_pivots = [[0, 0] for i in range(len(pivots))]

    for pivot in pivots:
        # create size-2 x 2 matrix Matrix[row][col]
        results = [0 for i in range(size)]
        largest_node_i = None
        largest_length = 0
        for i in range(size):
            if i in nodes:
                continue
            if largest_node_i is None:
                largest_node_i = i
            results[i] = 0
            for j in pivot:  # pivot is node_index
                results[i] += getLength(buf, i, j)
            if results[i] > largest_length:
                largest_length = results[i]  # store the latest largest
                largest_node_i = i  # store current node to be connect with
        #
        if largest_node_i is None:
            continue
        large_of_pivots[pivots.index(pivot)] = [largest_node_i, largest_length]

    node_to_join = None
    max_value = 0
    index_of_pivot = -1
    for i in range(len(large_of_pivots)):
        if large_of_pivots[i][1] == 0:
            continue
        if large_of_pivots[i][1] > max_value:
            max_value = large_of_pivots[i][1]
            node_to_join = large_of_pivots[i][0]
            index_of_pivot = i

    if node_to_join is None:
        print("No more nodes to join")
        return
    if index_of_pivot == -1:
        print("invalid condition")
        return
    print(
        "Join node",
        node_to_join + 1,
        "with pivot",
        index_of_pivot,
        [node + 1 for node in pivots[index_of_pivot]],
        "with value",
        max_value,
        "from pivot",
    )
    joinNode(node_to_join, pivots[index_of_pivot])
    return


while len(nodes) < size:
    findNext()


#  Summary
print("============")
print("Summary")
print("Nodes:", [node + 1 for node in nodes])


G = nx.DiGraph()
t = [1, 2, 7, 3, 6, 4, 5]
G.add_edges_from(
    [
        [1, 2],
        [2, 7],
        [7, 3],
        [3, 6],
        [6, 4],
        [4, 5],
        [5, 1],
    ],
    weight=0,
)

Gx.add_edges_from(brunch)

# can you write a function to add the edges?
print(brunch)

pos = nx.planar_layout(Gx)
nx.draw(
    Gx,
    pos,
    with_labels=True,
    font_weight="bold",
    node_color="lightblue",
)

plt.show()
