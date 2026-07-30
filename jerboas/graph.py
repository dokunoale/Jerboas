"""Graph: the data, and the entry point to the pipeline.

A node key in the source files is `<type>.<id>` -- so a node is really a (type,
id) pair, not an opaque string. The loader gives every type a contiguous block
of integers and stores a node as `start[type] + local`. Three things follow, and
they are the whole reason for the representation:

  * the universe is `range(N)`, so every per-node fact is a plain array indexed
    directly -- no dict, no hashing, no interning;
  * `keys_by_type` is a slice, not a stored partition;
  * a node constraint compiles to one boolean mask (see query._Compiler), which
    is what lets the engines drop their per-candidate predicate loop.

Relations are directed and stored once, as a single edge-labeled CSR plus its
transpose. Walking backwards reads the transpose instead of a separate `_r`
relation, so there is no `rv` flag, no duplicated edges, and `Edge()` traverses
both directions because both are always present.

Attributes are columnar and typed at load, so `year >= 1990` compares int64
against int64 instead of coercing a string once per candidate node. One of them
is virtual: `label` resolves per type to whichever column holds the readable
name, so a query can filter on it without knowing that movies call it `title`
and people call it `name`.
"""

import os
from array import array

import numpy as np
import scipy.sparse as sp

from .columns import build as build_column
from .keys import Key
from .query import Query

# a type's label column, in preference order; falling back to the first text
# column would pick `imdb` over `name` for a person, which is never what a
# caller printing a result wants
LABEL_PREFERENCE = ("name", "title", "label")

# the virtual attribute that resolves to whichever column above a type actually
# has. `movie.label` reads `title` and `person.label` reads `name`, so a caller
# can filter by the human-readable name without knowing the physical column --
# the same thing Key.label already gives them on the way out.
LABEL = "label"

INTERACT = "has_interact"


def _split(key):
    """'movie.123' -> ('movie', '123'). The id may be any token; only the first
    dot separates, so ids containing dots survive."""
    type_, _, raw = key.partition(".")
    return type_, raw


class Graph:
    def __init__(self, kg=None, ui=None, attrs=None, sf=None, labels=None):
        self.kg = kg
        self.ui = ui
        # score filter on ui interactions: (min, max) inclusive; keeps
        # interactions whose rating falls in range, e.g. (3, inf) keeps >= 3
        self.sf = sf
        self.attr_files = attrs or []
        self.labels = labels or {}       # type -> label column, overriding the heuristic

        self._cache = {}                 # name -> memoized value; safe for the graph's lifetime

        # --- id allocation, mutable only during _load ---
        # nodes get a provisional id in first-seen order; _finalize permutes
        # those into per-type blocks. Splitting `type.id` is deferred to that
        # pass so it runs once per distinct node rather than twice per edge.
        self._id = {}                    # source key -> provisional id
        self._keys = []                  # provisional id -> source key
        self.relations = []              # relation name by code
        self._relation_code = {}

        # edges as three parallel int arrays, in provisional ids
        self._e_src = array("i")
        self._e_rel = array("i")
        self._e_tgt = array("i")

        self._attr_rows = {}             # type -> (column names, {provisional id: values})

        self._load()
        self._finalize()

    # --- loading -------------------------------------------------------------

    def _intern(self, key):
        """The provisional id of a source key -- one dict lookup per endpoint."""
        index = self._id.get(key)
        if index is None:
            index = self._id[key] = len(self._keys)
            self._keys.append(key)
        return index

    def _relation(self, name):
        code = self._relation_code.get(name)
        if code is None:
            code = self._relation_code[name] = len(self.relations)
            self.relations.append(name)
        return code

    def _add_edge(self, source, relation, target):
        self._e_src.append(self._intern(source))
        self._e_rel.append(self._relation(relation))
        self._e_tgt.append(self._intern(target))

    def _load(self):
        if self.kg:
            with open(self.kg, "r") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    self._add_edge(parts[0], parts[1], parts[2])

        if self.ui:
            with open(self.ui, "r") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    if self.sf is not None:              # rating filter (interaction stays binary)
                        if len(parts) < 3:
                            continue
                        rating = float(parts[2])
                        if not (self.sf[0] <= rating <= self.sf[1]):
                            continue
                    self._add_edge(parts[0], INTERACT, parts[1])

        for path in self.attr_files:
            self._load_attrs(path)

    def _load_attrs(self, path):
        # type inferred from the filename suffix: '.../ml.movie' -> 'movie';
        # first column is the node id, the rest are named attributes
        type_ = os.path.basename(path).split(".")[-1]
        with open(path, "r") as f:
            columns = f.readline().rstrip("\n").split("\t")[1:]
            rows = {}
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not parts or parts[0] == "":
                    continue
                rows[self._intern(f"{type_}.{parts[0]}")] = parts[1:]
        known = self._attr_rows.get(type_)
        if known is None:
            self._attr_rows[type_] = (columns, rows)
        else:                                            # a second file for the same type
            known[0].extend(columns)
            for node, values in rows.items():
                known[1].setdefault(node, []).extend(values)

    # --- freezing: ids become contiguous, edges become CSR -------------------

    def _finalize(self):
        """Permute the provisional ids into per-type blocks, then freeze.

        Sorting by (type, first-seen) is what makes a type a contiguous range,
        and therefore `keys_by_type` a slice and every per-node fact an array."""
        self.n_nodes = len(self._keys)
        self.types, self._type_pos = [], {}
        tags = np.empty(self.n_nodes, dtype=np.int32)
        raws = []
        for index, key in enumerate(self._keys):         # once per node, not per edge
            type_, raw = _split(key)
            tag = self._type_pos.get(type_)
            if tag is None:
                tag = self._type_pos[type_] = len(self.types)
                self.types.append(type_)
            tags[index] = tag
            raws.append(raw)

        order = np.lexsort((np.arange(self.n_nodes), tags))   # by type, then first-seen
        self._new_of_old = np.empty(self.n_nodes, dtype=np.int64)
        self._new_of_old[order] = np.arange(self.n_nodes)

        self.start = np.zeros(len(self.types) + 1, dtype=np.int64)
        self.start[1:] = np.cumsum(np.bincount(tags, minlength=len(self.types)))
        self._type_tag_of = tags[order]

        # per-type raw ids and the reverse lookup, in the new block order
        self._raw = [[] for _ in self.types]
        self._local = [{} for _ in self.types]
        for old in order.tolist():
            table = self._local[tags[old]]
            block = self._raw[tags[old]]
            table[raws[old]] = len(block)
            block.append(raws[old])

        self._build_adjacency()
        self._build_columns()

        # only the by-name lookup tables outlive the load
        self._e_src = self._e_rel = self._e_tgt = None
        self._attr_rows = self._id = self._keys = self._new_of_old = None

    def _build_adjacency(self):
        """One edge-labeled CSR plus its transpose.

        Sorting each node's slice by relation is what makes both access patterns
        cheap from a single store: a wildcard `Edge()` is the whole slice, and a
        named relation is a searchsorted sub-range of it. Two arrays per
        direction, versus a dict of dicts of lists holding 2x the edges."""
        if len(self._e_rel) == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            empty_p = np.zeros(self.n_nodes + 1, dtype=np.int64)
            self.out_indptr, self.out_indices, self.out_rels = empty_p, empty_i, empty_i
            self.in_indptr, self.in_indices, self.in_rels = empty_p.copy(), empty_i, empty_i
            return

        # one fancy-index each turns every provisional edge id into its block id
        src = self._new_of_old[np.frombuffer(self._e_src, dtype=np.int32)]
        tgt = self._new_of_old[np.frombuffer(self._e_tgt, dtype=np.int32)]
        rel = np.frombuffer(self._e_rel, dtype=np.int32).astype(np.int64)

        self.out_indptr, self.out_indices, self.out_rels = self._csr(src, tgt, rel)
        self.in_indptr, self.in_indices, self.in_rels = self._csr(tgt, src, rel)

    def _csr(self, key, value, rel):
        order = np.lexsort((rel, key))               # by source, then by relation
        indptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(key, minlength=self.n_nodes), out=indptr[1:])
        return indptr, value[order], rel[order]

    def _build_columns(self):
        """Per type, {attribute: Column} indexed by local id -- plus `id`, which
        is a real column now instead of a split() on every read."""
        self.columns = {}
        self.label = {}
        for tag, type_ in enumerate(self.types):
            size = len(self._raw[tag])
            table = {"id": build_column(self._raw[tag])}
            names, rows = self._attr_rows.get(type_, ([], {}))
            # a row is keyed by provisional id; resolve it to a local one once,
            # not once per column
            placed = [(int(self._new_of_old[node]) - int(self.start[tag]), row)
                      for node, row in rows.items()]
            for position, name in enumerate(names):
                values = [None] * size
                for local, row in placed:
                    if position < len(row) and row[position] != "":
                        values[local] = row[position]
                table[name] = build_column(values)
            self.columns[type_] = table
            self.label[type_] = self._label_column(type_, table)

    def _label_column(self, type_, table):
        if type_ in self.labels:
            return self.labels[type_]
        for name in LABEL_PREFERENCE:
            if name in table:
                return name
        return None

    # --- node identity -------------------------------------------------------

    def type_of(self, index):
        return self.types[self._type_tag_of[index]]

    def block(self, type_):
        """The half-open index range of a type: `keys_by_type`, as arithmetic."""
        tag = self._type_pos.get(type_)
        if tag is None:
            return 0, 0
        return int(self.start[tag]), int(self.start[tag + 1])

    def local(self, index):
        return int(index) - int(self.start[self._type_tag_of[index]])

    def raw_id(self, index):
        return self.columns[self.type_of(index)]["id"].get(self.local(index))

    def label_of(self, index):
        value = self.value(index, LABEL)
        if value is not None:
            return value
        return f"{self.type_of(index)}.{self.raw_id(index)}"

    def attrs_of(self, index):
        local = self.local(index)
        return {name: column.get(local)
                for name, column in self.columns[self.type_of(index)].items()}

    def resolve(self, type_, name):
        """An attribute name as this type stores it. Only `label` is virtual; it
        is resolved here, in the one place both querying and rendering pass
        through, so the two can never disagree about what a label is."""
        return self.label.get(type_) if name == LABEL else name

    def value(self, index, name):
        """One attribute of one node, for rendering a single result cell."""
        type_ = self.type_of(index)
        column = self.columns[type_].get(self.resolve(type_, name))
        return None if column is None else column.get(self.local(index))

    def column(self, type_, name):
        """A whole typed attribute column, for building a mask in one go.

        None when the type has no such column -- including a type with no label
        at all (nothing in LABEL_PREFERENCE), which then simply matches nothing
        rather than falling back to some synthetic string."""
        table = self.columns.get(type_)
        return None if table is None else table.get(self.resolve(type_, name))

    def key(self, index):
        return Key(self, int(index))

    def lookup(self, spec):
        """A node index from a source key ('movie.123') or a (type, id) pair --
        the way a caller names a node the query did not hand them."""
        if isinstance(spec, Key):
            return int(spec)
        type_, raw = spec if isinstance(spec, tuple) else _split(spec)
        tag = self._type_pos.get(type_)
        if tag is None:
            return None
        local = self._local[tag].get(str(raw))
        return None if local is None else int(self.start[tag]) + local

    # --- traversal -----------------------------------------------------------

    def relation_code(self, name):
        return self._relation_code.get(name)

    def expand(self, index, relation=None, reverse=None):
        """Out-neighbours as [(target, code)], where a negative code means the
        edge was walked backwards.

        `reverse=None` with no named relation is the wildcard: both directions,
        because a directed store plus its transpose is exactly what the old `_r`
        duplication was faking."""
        out = []
        if reverse is not True:
            self._segment(out, self.out_indptr, self.out_indices, self.out_rels,
                          index, relation, False)
        if reverse is not False:
            self._segment(out, self.in_indptr, self.in_indices, self.in_rels,
                          index, relation, True)
        return out

    def _segment(self, out, indptr, indices, rels, index, relation, reverse):
        lo, hi = int(indptr[index]), int(indptr[index + 1])
        if lo == hi:
            return
        if relation is not None:
            # the slice is sorted by relation, so the named one is a sub-range;
            # both bounds are searched against the same original segment
            segment = rels[lo:hi]
            left = int(np.searchsorted(segment, relation, "left"))
            right = int(np.searchsorted(segment, relation, "right"))
            lo, hi = lo + left, lo + right
            if lo >= hi:
                return
        code = ~relation if (reverse and relation is not None) else relation
        targets = indices[lo:hi].tolist()
        if relation is not None:
            out.extend((t, code) for t in targets)
        else:
            codes = rels[lo:hi].tolist()
            out.extend(zip(targets, (~c for c in codes) if reverse else codes))

    def has_edge(self, source, target, relation=None, reverse=False):
        """Does this edge exist? The anti-join test -- a membership check against
        an array slice, so no neighbour list is materialized to answer it."""
        directions = ((self.out_indptr, self.out_indices, self.out_rels),) if reverse is False \
            else ((self.in_indptr, self.in_indices, self.in_rels),) if reverse is True \
            else ((self.out_indptr, self.out_indices, self.out_rels),
                  (self.in_indptr, self.in_indices, self.in_rels))
        for indptr, indices, rels in directions:
            lo, hi = int(indptr[source]), int(indptr[source + 1])
            if relation is not None:
                segment = rels[lo:hi]
                left = int(np.searchsorted(segment, relation, "left"))
                right = int(np.searchsorted(segment, relation, "right"))
                lo, hi = lo + left, lo + right
            if lo < hi and target in indices[lo:hi]:
                return True
        return False

    def neighbours(self, index):
        """Every neighbour, both directions, deduplicated -- the undirected
        one-hop set a reachability strategy walks."""
        out = self.out_indices[self.out_indptr[index]:self.out_indptr[index + 1]]
        into = self.in_indices[self.in_indptr[index]:self.in_indptr[index + 1]]
        return set(out.tolist()) | set(into.tolist())

    def degree(self, relation=None, reverse=False):
        """Per-node arity of a named relation, for every node at once -- a Degree
        predicate constrains all candidates in one comparison. A relation the
        graph never saw has arity zero everywhere."""
        return self.cached(("degree", relation, reverse), lambda: self._degree(relation, reverse))

    def _degree(self, relation, reverse):
        indptr = self.in_indptr if reverse else self.out_indptr
        if relation is None:
            return np.diff(indptr)
        code = self._relation_code.get(relation)
        if code is None:
            return np.zeros(self.n_nodes, dtype=np.int64)
        rels = self.in_rels if reverse else self.out_rels
        sources = np.repeat(np.arange(self.n_nodes), np.diff(indptr))
        return np.bincount(sources[rels == code], minlength=self.n_nodes)

    # --- sparse views the strategies rank with -------------------------------

    def adjacency(self):
        """The undirected adjacency as one N x N CSR, counting multi-edges.

        Both directions, so random-walk mass flows symmetrically -- the property
        `rv=True` used to buy by physically duplicating every edge."""
        return self.cached("adjacency", self._build_matrix)

    def _build_matrix(self):
        sources = np.repeat(np.arange(self.n_nodes), np.diff(self.out_indptr))
        both_src = np.concatenate([sources, self.out_indices])
        both_tgt = np.concatenate([self.out_indices, sources])
        data = np.ones(len(both_src))
        return sp.csr_matrix((data, (both_src, both_tgt)),
                             shape=(self.n_nodes, self.n_nodes))

    def relation_matrix(self, relation):
        """One relation as an N x N CSR, source -> target. Block-slice it to get
        e.g. the user-item matrix: `m[u0:u1, i0:i1]`."""
        return self.cached(("relation_matrix", relation),
                           lambda: self._build_relation_matrix(relation))

    def _build_relation_matrix(self, relation):
        code = self._relation_code.get(relation)
        sources = np.repeat(np.arange(self.n_nodes), np.diff(self.out_indptr))
        keep = slice(None) if code is None else (self.out_rels == code)
        rows, cols = sources[keep], self.out_indices[keep]
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(self.n_nodes, self.n_nodes))

    # --- pipeline entry point (the only method that starts a query) ----------

    def select(self, *projections):
        return Query(self, *projections)

    query = select   # alias

    # --- derived, memoized graph facts ---------------------------------------

    def cached(self, key, builder):
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def schema(self):
        """{type: {"columns": set, "relations": set}} -- the schema the Query
        consults to resolve column-vs-relation and label columns."""
        return self.cached("schema", self._build_schema)

    def _build_schema(self):
        schema = {}
        for tag, type_ in enumerate(self.types):
            lo, hi = int(self.start[tag]), int(self.start[tag + 1])
            relations = set()
            for indptr, rels in ((self.out_indptr, self.out_rels),
                                 (self.in_indptr, self.in_rels)):
                span = rels[int(indptr[lo]):int(indptr[hi])]
                relations.update(self.relations[c] for c in np.unique(span).tolist())
            columns = set(self.columns[type_])
            if self.label.get(type_) is not None:
                columns.add(LABEL)          # queryable, so the schema advertises it
            schema[type_] = {"columns": columns, "relations": relations}
        return schema
