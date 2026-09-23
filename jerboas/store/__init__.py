"""The data: what a graph holds, and what outlives a process.

    graph.py        Graph -- integer ids in typed blocks, an edge-labeled CSR
                    and its transpose, typed attribute columns
    columns.py      typed, nullable attribute columns
    keys.py         Key -- a node, outside the frame
    checkpoint.py   a trained model's arrays, rebound to a graph by name

Nothing here decides anything about a query: the graph holds everything and
answers what it is asked.
"""
