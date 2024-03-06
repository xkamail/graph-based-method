import matplotlib.pyplot as plt
import networkx as nx

# init graph

buf = []
assign = []

# size = int(input("how many nodes:"))
size = 5
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
buf = [
    [0, 9, 8, 10, 0],
    [0, 0, 12, 13, 7],
    [0, 0, 0, 20, 0],
    [0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0],
]


# 7x7 testdata
# buf = [
#     [0, 20, 0, 0, 2, 8, 12],
#     [0, 0, 0, 0, 2, 8, 12],
#     [0, 0, 0, 0, 3, 10, 18],
#     [0, 0, 0, 0, 3, 10, 18],
#     [0, 0, 0, 0, 0, 8, 7],
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0],
# ]

# show the matrix
# print(buf)

# for now we have to construct triangle with 3 nodes and have single center
if len(buf) < 3:
    print("Not enough nodes")
    exit()


def getLength(buf=[], i=0, j=0):
    v = buf[i][j]
    if v == 0:
        v = buf[j][i]
    return v


G = nx.Graph()  # graph for calculation not render

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


Gx = nx.Graph()

for i, j in brunch:
    Gx.add_edges_from(
        [
            (
                i,
                j,
            ),
        ],
        weight=getLength(buf, i - 1, j - 1),
    )


def calculate_triangular_positions(G):
    # Get the nodes of the graph
    nodes = list(Gx.nodes())

    # Define the positions for the nodes
    # The top node is at the top of the triangle
    # The base nodes are at the bottom
    pos = {
        nodes[0]: (0, 0),  # Top node
        nodes[1]: (1, 0),  # Base node 1
        nodes[2]: (0.5, 1),  # Base node 2
    }

    # Place node 3 in the center of the triangle formed by nodes 1, 2, and 7
    pos[nodes[3]] = (0.5, 0.5)  # Assuming node 3 is the fourth node

    # Find the nodes that are connected to all three nodes of the outer triangle
    outer_triangle_nodes = {nodes[0], nodes[1], nodes[2]}
    center_nodes = {nodes[3]}  # Start with node 3 as it's already placed in the center
    for node in nodes[4:]:
        # get nodes nightbors only frist 3 nodes
        neighbors = list(G.neighbors(node))
        # so we know that node is connect with 3 triangle
        # we have to calculate the position of the node
        # to stay inside three neighbors connected
        neighbor_positions = [pos[n] for n in neighbors[:3]]
        center_x = sum(x for x, y in neighbor_positions) / len(neighbor_positions)
        center_y = sum(y for x, y in neighbor_positions) / len(neighbor_positions)
        pos[node] = (center_x, center_y)

    return pos


pos = calculate_triangular_positions(Gx)

# Draw the graph
nx.draw_networkx_nodes(Gx, pos, node_size=500)
nx.draw_networkx_edges(Gx, pos, width=2)
nx.draw_networkx_labels(Gx, pos, font_size=12, font_color="black")

# draw weight on edges
labels = nx.get_edge_attributes(Gx, "weight")
nx.draw_networkx_edge_labels(Gx, pos, edge_labels=labels)

# Show the plot
plt.axis("off")
plt.show()
