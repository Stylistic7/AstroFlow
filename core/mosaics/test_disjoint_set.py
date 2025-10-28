import random

from mosaics.disjoint_set import DisjointSet

def test_singletons():
    d = DisjointSet(5)
    assert len(list(d.roots())) == 5
    for i in range(5):
        assert d.size(i) == 1
        assert d.same(i, i)

def test_union_connects_and_sizes():
    d = DisjointSet(6)
    d.union(0, 1)
    d.union(1, 2)
    d.union(3, 4)
    assert d.same(0, 2)
    assert not d.same(0, 3)
    assert d.size(0) == 3
    assert d.size(4) == 2
    assert d.size(5) == 1
    g = sorted(sorted(g) for g in d.groups())
    assert g == [[0,1,2], [3,4], [5]]

def test_idempotent_union():
    d = DisjointSet(4)
    assert d.union(0, 1) is True
    assert d.union(1, 0) is False
    assert d.union(0, 1) is False
    assert d.same(0, 1)

def test_path_compression_effect():
    d = DisjointSet(8)
    # create a chain by forcing unions in a way that likely makes a path
    for i in range(7):
        d.p[i+1] = i  # manually link to force a tall tree for the test
    before = d.p.copy()
    _ = d.find(7)                 # compress along the path
    after = d.p.copy()
    assert any(b != a for a, b in zip(before, after))

def test_from_edges_equivalence():
    edges = [(0,1),(2,3),(3,4),(5,6)]
    d = DisjointSet.from_edges(8, edges)
    assert d.same(2, 4)
    assert not d.same(1, 2)
    assert d.size(2) == 3
    assert d.size(7) == 1

def test_random_consistency(seed: int = 123, n: int = 200):
    rng = random.Random(seed)
    d = DisjointSet(n)
    # Naive connectivity via adjacency, then transitive closure
    parent = list(range(n))
    def find_naive(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union_naive(a,b):
        ra, rb = find_naive(a), find_naive(b)
        if ra != rb:
            parent[rb] = ra
    edges = []
    for _ in range(2000):
        a = rng.randrange(n); b = rng.randrange(n)
        edges.append((a,b))
        d.union(a, b)
        union_naive(a, b)
    # Compare components
    parts = {}
    for i in range(n):
        parts.setdefault(d.find(i), set()).add(i)
    parts = sorted(sorted(s) for s in parts.values())
    naive_parts = {}
    for i in range(n):
        naive_parts.setdefault(find_naive(i), set()).add(i)
    naive_parts = sorted(sorted(s) for s in naive_parts.values())
    assert parts == naive_parts