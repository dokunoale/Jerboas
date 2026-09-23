"""The query: a polars frame that knows which of its columns are nodes.

    frame.py      Frame -- hop, attrs, labels, like, and polars forwarded
    expr.py       v / col -- names, resolved late by the frame that has the graph
    resolve.py    what turns a name into a column; a hop that has not built its rows
    traverse.py   one hop, as a gather over CSR slices
"""
