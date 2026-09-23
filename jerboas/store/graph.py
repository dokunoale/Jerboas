"""Graph: the data the frames are a view of.

**An id is a position.** Ids are dense integers `0..n-1` within a type, so
`song.42` is stored at `start["song"] + 42` and `lookup` is arithmetic -- no
translation table, and no dependence on the order a file happens to list things
in. A file that says otherwise is refused at load, because the alternative is a
graph whose contents depend on how it was written. From that: the universe is
`range(N)`, so every per-node fact is a plain array; a type's block is a slice;
and a hop is a gather over CSR slices (traverse.py) rather than a join.

Relations are stored once, as an edge-labeled CSR plus its transpose, so walking
backwards reads the transpose rather than a duplicated `_r` relation.

Every edge carries a weight defaulting to 1.0 -- there is no unweighted edge,
only one whose weight nobody wrote down -- and what counts as good enough is a
question for the query, never for the loader.

Loading is columnar because what runs once per edge has to be C: a chunk becomes
columns with one `replace` and one `split`, and the Python that is left runs
once per distinct node. Attributes are typed at load, so reading one into a
frame is a gather and comparing it is one array operation.

`readable` is the one thing the graph is told rather than reads: which column a
person reads, per type. Nothing guesses it, and it is a fact about the dataset
rather than about a query.
"""

import os
from itertools import chain

import numpy as np
import polars as pl
import scipy.sparse as sp

from .columns import Column, build as build_column
from ..search.fuzzy import words_of
from ..query.frame import Frame, RELATION
from .keys import Key

# Every type has these two columns and they are generated, not read: `id` is the
# node's position, and `label` is what identifies it to anything outside the
# graph. Without `renumber` they are the same array under two names -- the
# position is the identity -- and with it, `label` holds the id the source used.
# Neither is virtual: `graph.column(type, name)` is a dict lookup, and a query
# that wants `label` to mean a readable column says so with `Node.alias`.
ID = "id"
LABEL = "label"

# the weight of an edge whose file gives it no score
DEFAULT_WEIGHT = 1.0

# how much of an edge file becomes columns at a time. Splitting the whole file
# at once would hold one Python string per *field* until the load finished --
# millions of them, costing more than the graph they describe. Between 1 KB and
# 32 MB the time is flat and only the memory moves, so the size is chosen for
# the memory.
CHUNK = 1 << 16


def _split(key):
    """'movie.123' -> ('movie', '123'). Only the first dot separates, so a type
    name holding one would still split where it should."""
    type_, _, raw = key.partition(".")
    return type_, raw


# --- reading a file by column rather than by line ---------------------------

def _chunks(handle, first):
    """The file as whole-line chunks, `first` being the line already read off it
    to learn the shape."""
    remainder = first
    while True:
        block = handle.read(CHUNK)
        if not block:
            break
        block = remainder + block
        cut = block.rfind("\n")
        if cut < 0:                      # a line longer than the chunk: keep reading
            remainder = block
            continue
        remainder, block = block[cut + 1:], block[:cut + 1]
        yield block
    if remainder:
        yield remainder if remainder.endswith("\n") else remainder + "\n"


def _table(path, header, minimum):
    """An edge file as parallel columns of strings, a chunk at a time.

    The shape is read off the first data line, not the header: what the file
    holds is decided by what is in it. A chunk whose lines are not all that wide
    falls back to `_ragged` -- the fast path needs a rectangle, and a file is
    allowed not to be one.

    `minimum` is how many columns make a row worth reading: three for a triple,
    two for an edge whose relation the filename already gave.
    """
    with open(path, "r") as handle:
        first = handle.readline()
        if header:
            first = handle.readline()
        if not first:
            return
        width = first.count("\t") + 1
        if width < minimum:
            return
        for block in _chunks(handle, first):
            yield _rectangle(block, width) or _ragged(block, width, minimum)


def _rectangle(block, width):
    """Columns from a chunk of equal-width lines, or None when it is not one.

    Two separators and one pass: making the newline a tab as well leaves a flat
    field list that strides into columns, where the obvious reading makes two
    Python calls per line."""
    fields = block.replace("\n", "\t").split("\t")
    del fields[-1]                       # the chunk ends on a newline
    if not fields or len(fields) % width:
        return None
    return [fields[index::width] for index in range(width)]


def _ragged(block, width, minimum):
    """Columns from a chunk whose lines vary: a line too short to be an edge is
    dropped, and a missing trailing cell reads as absent."""
    columns = [[] for _ in range(width)]
    for line in block.split("\n"):
        parts = line.split("\t")
        if len(parts) < minimum or parts[0] == "":
            continue
        for index, column in enumerate(columns):
            column.append(parts[index] if index < len(parts) else "")
    return columns


def _position(key, raw):
    """An id as the position it names, or a refusal.

    Loading a file that does not follow the format is a bug in whatever wrote
    it, so it raises here rather than being accommodated. `lookup` answers the
    different question -- "is there such a node?" -- and answers it with None."""
    if not raw.isdigit():
        raise ValueError(f"{key!r}: a node id is a position, so it must be a "
                         f"non-negative integer")
    return int(raw)


def _sortable(raw):
    """A source id as something to sort by: numbers as numbers and before text,
    so a renumbering keeps `1 … 1682` in the order anyone would expect."""
    return (0, int(raw), "") if raw.isdigit() else (1, 0, raw)


def _drain(parts):
    """The staged chunks as one array, letting go of the chunks themselves."""
    joined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    parts.clear()
    return joined


def _index(column):
    """{word: sorted local ids}."""
    postings = {}
    for local in range(len(column)):
        value = column.get(local)
        if value is None:
            continue
        for word in set(words_of(value)):
            postings.setdefault(word, []).append(local)
    return {word: np.asarray(rows, dtype=np.int32)
            for word, rows in postings.items()}


def _matrix(series):
    """A column of lists as one (n, d) float32 block."""
    values = series.to_numpy()
    if values.dtype == object:                    # ragged or lazily typed
        values = np.stack([np.asarray(one, dtype=np.float32) for one in values])
    return np.ascontiguousarray(values, dtype=np.float32)


def _normalize(block):
    """Rows scaled to unit length; a zero row stays zero rather than becoming a
    direction it never had."""
    lengths = np.linalg.norm(block, axis=1, keepdims=True)
    return block / np.where(lengths > 0, lengths, 1.0)


def _edge_frames(edges):
    """(relation or None, frame) pairs, however the edges were given."""
    if isinstance(edges, dict):
        return list(edges.items())
    if isinstance(edges, (list, tuple)):
        return [(None, one) for one in edges]
    return [(None, edges)]


def _node_keys(frame, spec):
    """A column of source keys, from a column that holds them or from a column
    of ids and the type they belong to.

    The concatenation runs in polars rather than in Python: it is the one thing
    here that happens once per edge."""
    if isinstance(spec, str):
        if spec not in frame.columns:
            raise ValueError(f"no column {spec!r} in these edges; it has: "
                             f"{', '.join(frame.columns)}")
        return frame[spec].cast(pl.String).to_list()
    type_, column = spec
    if column not in frame.columns:
        raise ValueError(f"no column {column!r} in these edges; it has: "
                         f"{', '.join(frame.columns)}")
    return frame.select(
        pl.concat_str([pl.lit(f"{type_}."), pl.col(column).cast(pl.String)])
    ).to_series().to_list()


def _weights(column):
    """A score column as floats. An empty cell is an edge nobody scored, which
    is not the same as an edge scored zero."""
    try:
        return np.array(column, dtype=np.float64)
    except ValueError:
        return np.array([value or DEFAULT_WEIGHT for value in column], dtype=np.float64)


class Graph:
    def __init__(self, kg=None, edges=None, attrs=None, renumber=False, readable=None):
        self.kg = kg
        self.edge_files = edges or []
        self.attr_files = attrs or []
        self._prepare(renumber, readable)
        self._load()
        self._finalize()

    def _prepare(self, renumber, readable):
        """The mutable state a build needs, whether it reads files or frames."""
        self.renumber = renumber         # ids are not positions: assign them here
        # {type: column a person reads}. Declared, never inferred -- see labels()
        self.readable = dict(readable or {})

        self._cache = {}                 # name -> memoized value; safe for the graph's lifetime

        # --- id allocation, mutable only during _load ---
        # nodes get a provisional id in first-seen order; _finalize permutes
        # those into per-type blocks. Splitting `type.id` is deferred to that
        # pass so it runs once per distinct node rather than twice per edge.
        self._id = {}                    # source key -> provisional id
        self._keys = []                  # provisional id -> source key
        self.relations = []              # relation name by code
        self._relation_code = {}

        # edges as four parallel columns, in provisional ids: one array per
        # chunk read, concatenated once in _finalize
        self._e_src = []
        self._e_rel = []
        self._e_tgt = []
        self._e_weight = []

        self._attr_rows = {}             # type -> (column names, {provisional id: values})
        # type -> {name: (provisional ids, matrix)}. A vector is not a column:
        # it has no value a row can print and no order to sort by, and what a
        # query asks of one is nearness. Kept as one (n, d) float32 block per
        # type, so scoring a candidate set is a matmul and not a gather of lists
        self._vector_rows = {}
        # a frame build sets neither, and _load reads both as "nothing to read"
        self.kg = getattr(self, "kg", None)
        self.edge_files = getattr(self, "edge_files", [])
        self.attr_files = getattr(self, "attr_files", [])

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

    def _register(self, sources, targets):
        """Intern every node key these edges mention.

        Order is irrelevant -- a node's place comes from its id, not from when
        it was first seen -- so this is only about touching each distinct key
        once. `dict.fromkeys` does the deduplication in C, leaving the loop to
        run once per distinct node rather than once per endpoint."""
        table, known = self._id, self._keys
        for key in dict.fromkeys(chain(sources, targets)):
            if key not in table:
                table[key] = len(known)
                known.append(key)

    def _ids_of(self, keys):
        """A column of source keys as the ids _register just handed out -- the
        lookup itself runs in C, one dict access per endpoint and no more."""
        return np.fromiter(map(self._id.__getitem__, keys), dtype=np.int32, count=len(keys))

    def _codes_of(self, names):
        """The same, for a column of relation names."""
        for name in dict.fromkeys(names):
            self._relation(name)
        return np.fromiter(map(self._relation_code.__getitem__, names),
                           dtype=np.int32, count=len(names))

    def _stage(self, sources, codes, targets, weights):
        self._register(sources, targets)
        self._e_src.append(self._ids_of(sources))
        self._e_rel.append(codes)
        self._e_tgt.append(self._ids_of(targets))
        self._e_weight.append(np.ones(len(sources)) if weights is None else weights)

    # --- building from frames rather than from files -------------------------

    @classmethod
    def from_frames(cls, edges, attrs=None, *, source="source", target="target",
                    relation="relation", score="score", renumber=False,
                    readable=None):
        """A graph out of whatever polars can read.

            Graph.from_frames({"has_interact": pl.read_parquet("ratings.parquet")},
                              attrs={"movie": movies},
                              source=("user", "user_id"),
                              target=("movie", "movie_id"))

        `edges` is one frame carrying a relation column, or a mapping from
        relation to a frame that needs none; `attrs` maps a type to a frame
        whose first column is the id.

        A column names nodes either as source keys (`movie.123`, the format the
        files use) or as a `(type, column)` pair -- a column of ids and the type
        they belong to, which is what data from anywhere else looks like.

        Everything else is the file loader's, ids-are-positions included, and it
        takes what it needs in one call because a graph is complete when it
        exists.
        """
        graph = cls.__new__(cls)
        graph._prepare(renumber, readable)
        for name, frame in _edge_frames(edges):
            graph._stage_frame(frame, name, source, target, relation, score)
        for type_, frame in (attrs or {}).items():
            graph._stage_attrs(frame, type_)
        graph._finalize()
        return graph

    def _stage_frame(self, frame, name, source, target, relation, score):
        """One frame of edges, as the four columns the loader stages."""
        frame = frame.pl if hasattr(frame, "pl") else frame
        sources = _node_keys(frame, source)
        targets = _node_keys(frame, target)
        if name is not None:
            codes = np.full(len(sources), self._relation(name), dtype=np.int32)
        elif relation in frame.columns:
            codes = self._codes_of(frame[relation].to_list())
        else:
            raise ValueError(
                f"these edges name no relation: give `edges` as a mapping from "
                f"relation to frame, or add a {relation!r} column")
        weights = (frame[score].to_numpy().astype(np.float64)
                   if score in frame.columns else None)
        self._stage(sources, codes, targets, weights)

    def _stage_attrs(self, frame, type_):
        """One frame of attributes: the id column, then the rest by name.

        A column of lists is a vector rather than an attribute -- the dtype
        says which, so nothing has to be declared twice."""
        frame = frame.pl if hasattr(frame, "pl") else frame
        identity = ID if ID in frame.columns else frame.columns[0]
        vectors = [name for name, dtype in frame.schema.items()
                   if name != identity and dtype.base_type() in (pl.List, pl.Array)]
        names = [name for name in frame.columns
                 if name != identity and name not in vectors]
        keys = [f"{type_}.{one}" for one in frame[identity].to_list()]
        if vectors:
            nodes = np.fromiter((self._intern(key) for key in keys),
                                dtype=np.int64, count=len(keys))
            block = self._vector_rows.setdefault(type_, {})
            for name in vectors:
                block[name] = (nodes, _matrix(frame[name]))
        values = [[None if one is None else str(one) for one in frame[name].to_list()]
                  for name in names]
        rows = {self._intern(key): [column[row] for column in values]
                for row, key in enumerate(keys)}
        known = self._attr_rows.get(type_)
        if known is None:
            self._attr_rows[type_] = (names, rows)
        else:                                            # a second frame for the type
            known[0].extend(names)
            for node, row in rows.items():
                known[1].setdefault(node, []).extend(row)

    # --- loading from files --------------------------------------------------

    def _load(self):
        if self.kg:
            self._load_kg(self.kg)

        for path in self.edge_files:
            self._load_edges(path)

        for path in self.attr_files:
            self._load_attrs(path)

    def _load_kg(self, path):
        """head <TAB> relation <TAB> tail, and optionally a score: the file that
        holds several relations at once, so the relation is a column."""
        for columns in _table(path, header=False, minimum=3):
            self._stage(columns[0], self._codes_of(columns[1]), columns[2],
                        _weights(columns[3]) if len(columns) > 3 else None)

    def _load_edges(self, path):
        """One relation per file, named by the filename suffix: '.../ml.has_interact'
        -> 'has_interact', the same rule _load_attrs reads a type by.

        A header row names the columns (source, target, score) so the file says
        what it holds; the score is the third column and may be left out
        entirely, in which case every edge weighs DEFAULT_WEIGHT."""
        relation = os.path.basename(path).split(".")[-1]
        for columns in _table(path, header=True, minimum=2):
            # named here rather than above, so an empty file leaves behind no
            # relation the graph has never seen an edge of
            code = np.full(len(columns[0]), self._relation(relation), dtype=np.int32)
            self._stage(columns[0], code, columns[1],
                        _weights(columns[2]) if len(columns) > 2 else None)

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
        """Put every node where its id says it goes, then freeze.

        A type's block is as long as its largest id, and a node sits at
        `start[type] + id`. There is no permutation and no ordering to preserve:
        two files listing the same graph in any two orders load to the same
        integers, which is what makes an id something a checkpoint, a URL or
        another dataset can refer to."""
        count = len(self._keys)
        self.types, self._type_pos = [], {}
        tags = np.empty(count, dtype=np.int32)
        places = np.empty(count, dtype=np.int64)
        raws = [] if self.renumber else None
        for index, key in enumerate(self._keys):         # once per node, not per edge
            type_, raw = _split(key)
            tag = self._type_pos.get(type_)
            if tag is None:
                tag = self._type_pos[type_] = len(self.types)
                self.types.append(type_)
            tags[index] = tag
            if raws is None:
                places[index] = _position(key, raw)
            else:
                raws.append(raw)

        sources = None if raws is None else self._renumber(tags, raws, places)

        sizes = np.zeros(len(self.types), dtype=np.int64)
        np.maximum.at(sizes, tags, places + 1)
        self.start = np.zeros(len(self.types) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.start[1:])
        self.n_nodes = int(self.start[-1])
        self._type_tag_of = np.repeat(np.arange(len(self.types), dtype=np.int32), sizes)

        # int32, and so every endpoint column downstream: a node id indexes an
        # array, and an array of two billion is not what runs out first
        self._new_of_old = (self.start[tags] + places).astype(np.int32)
        self._check_positions(count)

        self._build_adjacency()
        self._build_columns(sizes, sources)
        self._build_vectors(sizes)

        # only the by-name lookup tables outlive the load
        self._e_src = self._e_rel = self._e_tgt = self._e_weight = None
        self._attr_rows = self._id = self._keys = self._new_of_old = None
        self._vector_rows = None

    def _renumber(self, tags, raws, places):
        """Positions for a dataset whose ids are not positions, and the ids it
        used, per type, in the new order.

        Sorting each type's tokens is what makes the numbering a function of the
        node *set*: which file mentioned a node first, and how much of a file was
        read at a time, stop being able to move it. Numeric tokens sort as
        numbers and before text, so `movie.1 … movie.1682` keeps its order.

        The token itself becomes the `label` column, because it is the only
        thing a renumbered graph holds that anything outside it can name -- and
        a checkpoint fitted here must key on it (`alias="label"`), since the
        positions move whenever the node set does."""
        sources = [[] for _ in self.types]
        for tag in range(len(self.types)):
            rows = np.flatnonzero(tags == tag).tolist()
            rows.sort(key=lambda index: _sortable(raws[index]))
            for position, index in enumerate(rows):
                places[index] = position
                sources[tag].append(raws[index])
        return sources

    def _check_positions(self, count):
        """Every place in every block belongs to exactly one node.

        A skipped id leaves a node with no edges, no attributes and no way to
        tell it from one whose data went missing; a repeated one ('movie.7' and
        'movie.007') puts two nodes in one place and loses whichever loaded
        first. Both are bugs in whatever wrote the file, and neither is
        discoverable once the graph is built -- so they are refused here."""
        seen = np.zeros(self.n_nodes, dtype=bool)
        seen[self._new_of_old] = True
        if not seen.all():
            self._refuse(int(np.flatnonzero(~seen)[0]), "is never mentioned")
        if count != self.n_nodes:
            twice = np.flatnonzero(np.bincount(self._new_of_old,
                                               minlength=self.n_nodes) > 1)
            self._refuse(int(twice[0]), "is named twice")

    def _refuse(self, index, complaint):
        tag = int(np.searchsorted(self.start, index, "right")) - 1
        raise ValueError(f"{self.types[tag]}.{index - int(self.start[tag])} {complaint}: "
                         f"a node id is its position, so a type's ids are 0..n-1")

    def _build_adjacency(self):
        """One edge-labeled CSR plus its transpose.

        Sorting each node's slice by relation is what makes both access patterns
        cheap from a single store: a wildcard hop is the whole slice, and a
        named relation is a searchsorted sub-range of it. Two arrays per
        direction, versus a dict of dicts of lists holding 2x the edges.

        The weight rides along as a third array per direction, permuted by the
        same order: `out_weights[p]` is the weight of the edge ending at
        `out_indices[p]`, so a weight predicate is a slice like everything else
        here."""
        if not any(len(part) for part in self._e_rel):
            empty_i = np.zeros(0, dtype=np.int32)
            empty_w = np.zeros(0, dtype=np.float64)
            empty_p = np.zeros(self.n_nodes + 1, dtype=np.int64)
            self.out_indptr, self.out_indices = empty_p, empty_i
            self.out_rels, self.out_weights = empty_i, empty_w
            self.in_indptr, self.in_indices = empty_p.copy(), empty_i
            self.in_rels, self.in_weights = empty_i, empty_w.copy()
            return

        # one fancy-index each turns every provisional edge id into its block
        # id. Draining as it goes because the two CSR passes below allocate
        # twice what these columns hold, and holding both at once is the peak
        src = self._new_of_old[_drain(self._e_src)]
        tgt = self._new_of_old[_drain(self._e_tgt)]
        rel = _drain(self._e_rel)
        weight = _drain(self._e_weight)

        self.out_indptr, self.out_indices, self.out_rels, self.out_weights = \
            self._csr(src, tgt, rel, weight)
        self.in_indptr, self.in_indices, self.in_rels, self.in_weights = \
            self._csr(tgt, src, rel, weight)

    def _csr(self, key, value, rel, weight):
        order = np.lexsort((rel, key))               # by source, then by relation
        indptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(key, minlength=self.n_nodes), out=indptr[1:])
        return indptr, value[order], rel[order], weight[order]

    def _build_vectors(self, sizes):
        """Per type, {name: (size, d) float32} laid out by local id.

        A node the frame had no row for keeps a zero vector, which is nearest to
        nothing -- absence reads as "no answer", never as "here is one"."""
        self.vectors = {}
        for tag, type_ in enumerate(self.types):
            staged = self._vector_rows.get(type_)
            if not staged:
                continue
            low = int(self.start[tag])
            table = {}
            for name, (nodes, matrix) in staged.items():
                block = np.zeros((int(sizes[tag]), matrix.shape[1]), dtype=np.float32)
                block[self._new_of_old[nodes] - low] = matrix
                table[name] = block
            self.vectors[type_] = table

    def vector(self, type_, name):
        """One vector column, or None when the type has no such thing."""
        return self.vectors.get(type_, {}).get(name)

    def words(self, type_, name):
        """An inverted index over a text column: {word: the local ids holding it}.

        Built once per column and memoized, the way a degree or a normalized
        weight is. It answers the question a search actually asks -- which rows
        hold this word -- instead of walking every value to find out, and it
        answers it by *word*, which is the difference between `Toxic` matching
        `Toxicity` and not.

        None when the column is not text, since there is nothing to tokenize."""
        column = self.column(type_, name)
        if column is None or column.values.dtype != object:
            return None
        return self.cached(("words", type_, name), lambda: _index(column))

    def unit(self, type_, name):
        """The same block with every row scaled to length one, so nearness is a
        matmul. Memoized: normalising is O(n d) and the block never changes."""
        block = self.vector(type_, name)
        if block is None:
            return None
        return self.cached(("unit", type_, name), lambda: _normalize(block))

    def _build_columns(self, sizes, sources):
        """Per type, {attribute: Column} indexed by local id -- plus `id` and
        `label`, which are generated rather than read.

        Without `renumber` the two are the same Column object under two names,
        because the position *is* the identity and a second copy of it would
        only be a second thing to keep in step."""
        self.columns = {}
        for tag, type_ in enumerate(self.types):
            size = int(sizes[tag])
            identity = Column(np.arange(size, dtype=np.int64))
            table = {ID: identity,
                     LABEL: identity if sources is None else build_column(sources[tag])}
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
        return self.local(index)          # an id is a position; they are the same number

    def label_of(self, index):
        """What identifies this node outside the graph: the id the source gave
        it under `renumber`, and its position otherwise."""
        return self.value(index, LABEL)

    def attrs_of(self, index):
        local = self.local(index)
        return {name: column.get(local)
                for name, column in self.columns[self.type_of(index)].items()}

    def value(self, index, name):
        """One attribute of one node, for rendering a single result cell."""
        column = self.columns[self.type_of(index)].get(name)
        return None if column is None else column.get(self.local(index))

    def column(self, type_, name):
        """A whole typed attribute column, for building a mask in one go.

        None when the type has no such column, which then simply matches nothing
        rather than falling back to some synthetic string."""
        table = self.columns.get(type_)
        return None if table is None else table.get(name)

    def key(self, index):
        return Key(self, int(index))

    # --- the container protocol: a graph holds nodes -------------------------

    def __repr__(self):
        edges = len(self.out_indices)
        return (f"<Graph: {self.n_nodes} nodes in {len(self.types)} types "
                f"({', '.join(self.types)}), {edges} edges in "
                f"{len(self.relations)} relations ({', '.join(self.relations)})>")

    def __len__(self):
        return self.n_nodes

    def __iter__(self):
        return (Key(self, index) for index in range(self.n_nodes))

    def __contains__(self, spec):
        return self.lookup(spec) is not None

    def __getitem__(self, spec):
        """`g["movie.12"]` -- lookup by name, raising like any other mapping.

        `lookup` answers None for a node that is not there, which is what a
        caller sweeping a list of guesses wants; this is for the caller who
        believes the node exists and should hear about it if not."""
        index = self.lookup(spec)
        if index is None:
            raise KeyError(spec)
        return Key(self, index)

    def lookup(self, spec):
        """A node index from a source key ('movie.123') or a (type, id) pair --
        the way a caller names a node the query did not hand them.

        Arithmetic, because an id is a position: nothing to build before a name
        can be resolved, and nothing that can go stale. Asking for a node that
        is not there is a question, not an error, so it answers None -- unlike
        `_position`, which refuses a malformed *file*."""
        if isinstance(spec, Key):
            return int(spec)
        type_, raw = spec if isinstance(spec, tuple) else _split(spec)
        tag = self._type_pos.get(type_)
        if tag is None:
            return None
        try:
            position = int(raw)
        except (TypeError, ValueError):
            return None
        start, stop = int(self.start[tag]), int(self.start[tag + 1])
        return start + position if 0 <= position < stop - start else None

    # --- traversal -----------------------------------------------------------

    def relation_code(self, name):
        return self._relation_code.get(name)

    def target_types(self, relation=None, reverse=None):
        """The type a relation lands in, when it lands in only one.

        A schema fact rather than a fact about any query: read once per relation
        off the store and memoized. It is what lets a walk be described before it
        is taken -- `optimize` defers the walk, and something still has to know
        what the column it will fill is going to hold."""
        return self.cached(("target_types", relation, reverse),
                           lambda: self._target_types(relation, reverse))

    def _target_types(self, relation, reverse):
        code = None if relation is None else self._relation_code.get(relation)
        if relation is not None and code is None:
            return None
        found = set()
        for backwards in ((False, True) if reverse is None else (reverse,)):
            rels = self.in_rels if backwards else self.out_rels
            indices = self.in_indices if backwards else self.out_indices
            targets = indices if code is None else indices[rels == code]
            if len(targets):
                found.update(np.unique(self._type_tag_of[targets]).tolist())
        return self.types[found.pop()] if len(found) == 1 else None

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

    def sources(self, reverse=False):
        """The source node of every stored edge, expanded from the CSR row
        offsets: `indptr` says where each node's block starts, this says which
        node each position belongs to.

        int32 because it holds node ids, and memoized because a relation's hop
        bounds are counted from it -- on the full Spotify graph this array is
        283 MB, and rebuilding it per relation is what that would otherwise
        cost."""
        return self.cached(("sources", reverse), lambda: self._expand_sources(reverse))

    def _expand_sources(self, reverse):
        indptr = self.in_indptr if reverse else self.out_indptr
        return np.repeat(np.arange(self.n_nodes, dtype=np.int32), np.diff(indptr))

    def _degree(self, relation, reverse):
        indptr = self.in_indptr if reverse else self.out_indptr
        if relation is None:
            return np.diff(indptr)
        code = self._relation_code.get(relation)
        if code is None:
            return np.zeros(self.n_nodes, dtype=np.int64)
        rels = self.in_rels if reverse else self.out_rels
        return np.bincount(self.sources(reverse)[rels == code], minlength=self.n_nodes)

    # --- edge weights --------------------------------------------------------

    def weight_bounds(self):
        """The lowest and highest stored weight of each relation, as two arrays
        indexed by relation code -- what puts a traversed weight on its own
        [0, 1] scale without comparing it to another relation's."""
        return self.cached("weight_bounds", self._weight_bounds)

    def _weight_bounds(self):
        count = max(len(self.relations), 1)
        low, high = np.full(count, np.inf), np.full(count, -np.inf)
        # one masked pass per relation rather than `np.minimum.at`, which is
        # numpy's unbuffered fallback: 3x on 2.78M edges and three relations.
        # Relations stay few, so a handful of extra passes is the cheap side
        for code in range(len(self.relations)):
            weights = self.out_weights[self.out_rels == code]
            if weights.size:
                low[code], high[code] = weights.min(), weights.max()
        return low, high

    def weights(self, normalized=False):
        """The stored weights as an (out, in) pair, aligned with out_indices /
        in_indices."""
        if not normalized:
            return self.out_weights, self.in_weights
        return self.cached("norm_weights", self._normalize_weights)

    def _normalize_weights(self):
        """Min-max into [0, 1], per relation.

        Per relation because scales do not compare across them: a 1-5 rating and
        a cosine similarity are both floats and mean nothing to each other. A
        relation whose weights are all equal -- every edge that was never given a
        score -- maps to 1.0 rather than 0.0, so a graph with no weights behaves
        exactly as one that never had the notion.

        This is what makes an unbounded score usable by the parts that must
        combine or accumulate one: a walk needing non-negative transitions, a
        loss weighting its examples."""
        low, high = self.weight_bounds()
        span = high - low

        def scale(rels, weights):
            scaled = np.ones(len(weights))
            varying = span[rels] > 0
            codes = rels[varying]
            scaled[varying] = (weights[varying] - low[codes]) / span[codes]
            return scaled

        return scale(self.out_rels, self.out_weights), scale(self.in_rels, self.in_weights)

    # --- sparse views the strategies rank with -------------------------------

    def adjacency(self, weights=None):
        """The undirected adjacency as one N x N CSR, counting multi-edges.

        Both directions, so random-walk mass flows symmetrically -- the property
        `rv=True` used to buy by physically duplicating every edge.

        `weights` picks what fills the cells: None counts an edge as 1, "raw"
        uses its stored score, "norm" the per-relation min-max of it. A random
        walk wants "norm" and not "raw": a negative score is not a transition."""
        return self.cached(("adjacency", weights), lambda: self._build_matrix(weights))

    def _build_matrix(self, weights):
        sources = self.sources()
        both_src = np.concatenate([sources, self.out_indices])
        both_tgt = np.concatenate([self.out_indices, sources])
        data = self._edge_data(weights)
        return sp.csr_matrix((np.concatenate([data, data]), (both_src, both_tgt)),
                             shape=(self.n_nodes, self.n_nodes))

    def relation_matrix(self, relation, weights=None):
        """One relation as an N x N CSR, source -> target. Block-slice it to get
        e.g. the user-item matrix: `m[u0:u1, i0:i1]`. `weights` reads as it does
        for adjacency()."""
        return self.cached(("relation_matrix", relation, weights),
                           lambda: self._build_relation_matrix(relation, weights))

    def _build_relation_matrix(self, relation, weights):
        sources = self.sources()
        if relation is None:                       # every relation
            keep = slice(None)
        else:
            code = self._relation_code.get(relation)
            # a name the graph never saw matches nothing. Reading it as "all of
            # them" is how a typo used to become the whole graph
            keep = (self.out_rels == code) if code is not None \
                else np.zeros(len(sources), dtype=bool)
        rows, cols = sources[keep], self.out_indices[keep]
        return sp.csr_matrix((self._edge_data(weights)[keep], (rows, cols)),
                             shape=(self.n_nodes, self.n_nodes))

    def _edge_data(self, weights):
        """What one stored edge contributes to a matrix cell: its existence, its
        score, or its normalized score."""
        if weights is None:
            return np.ones(len(self.out_indices))
        return self.weights(normalized=(weights == "norm"))[0]

    # --- the frame: where a query starts -------------------------------------

    def nodes(self, *positional, **named):
        """A column of nodes, as a frame.

            g.nodes()                     every node, with its type
            g.nodes("movie")              every movie, in a column called `movie`
            g.nodes(rec="movie")          the same, under the name the query uses
            g.nodes(seed=keys)            the nodes someone named

        One variable per call, on purpose: two would be a cross product, and a
        cross product is never what was meant. Columns meet each other by
        hopping between them or by joining, both of which say so."""
        if len(positional) + len(named) > 1:
            raise TypeError(
                "nodes(...) makes one column at a time: two would be a cross "
                "product. Reach the second with hop(...), or join two frames.")
        if positional:
            name = value = positional[0]
        elif named:
            (name, value), = named.items()
        else:
            return self._all_nodes()

        if isinstance(value, str):
            if value not in self._type_pos:
                raise ValueError(f"no type {value!r} in this graph; it has: "
                                 f"{', '.join(sorted(self.types))}")
            low, high = self.block(value)
            ids = np.arange(low, high, dtype=np.int32)
            return Frame(self, {name: ids}, {name: value})

        ids = self.ids_of(value)
        types = {self.type_of(index) for index in ids.tolist()}
        single = types.pop() if len(types) == 1 else None
        return Frame(self, {name: ids}, {name: single})

    def _all_nodes(self):
        return Frame(self, {"node": np.arange(self.n_nodes, dtype=np.int32),
                            "type": [self.types[tag] for tag in self._type_tag_of.tolist()]},
                     {"node": None})

    def edges(self, relation=None, normalized=False):
        """Every stored edge, or every edge of one relation, as a frame.

        The CSR is already this table -- source, relation, target, weight -- so
        this is a view of it rather than a copy of it. It is what a query about
        the edges themselves asks, and what a training run is handed to say which
        of them it may learn from.

        The relation rides along as an Enum: dictionary-encoded, so naming it on
        seventy million rows costs a byte each rather than a string each."""
        sources = self.sources()
        keep = slice(None)
        if relation is not None:
            code = self._relation_code.get(relation)
            keep = (np.zeros(len(sources), dtype=bool) if code is None
                    else self.out_rels == code)
        names = pl.Enum(self.relations) if self.relations else pl.String
        data = pl.DataFrame({
            "source": sources[keep],
            RELATION: pl.Series([self.relations[code]
                                 for code in self.out_rels[keep].tolist()], dtype=names),
            "target": self.out_indices[keep],
            "score": self.weights(normalized)[0][keep],
        })
        return Frame(self, data, {"source": None, "target": None})

    def ids_of(self, values):
        """A set of nodes as an int32 array, however it was named: Keys, source
        strings, (type, id) pairs, integers, or a frame's node column."""
        if hasattr(values, "ids"):                       # a Frame
            return values.ids()
        if isinstance(values, np.ndarray):
            return values.astype(np.int32, copy=False)
        out = []
        for value in values:
            index = self.lookup(value) if not isinstance(value, (int, np.integer)) else int(value)
            if index is not None:
                out.append(index)
        return np.asarray(out, dtype=np.int32)

    # --- derived, memoized graph facts ---------------------------------------

    def cached(self, key, builder):
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def schema(self):
        """{type: {"columns": set, "relations": set}} -- what this graph holds,
        for a caller who wants to look before asking."""
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
            schema[type_] = {"columns": set(self.columns[type_]), "relations": relations}
        return schema
