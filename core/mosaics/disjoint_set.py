# dsu.py
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Iterator, List

class DisjointSet:
    __slots__ = ("p", "sz")

    def __init__(self, n: int):
        self.p: List[int] = list(range(n))
        self.sz: List[int] = [1] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]
        return True

    def same(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def size(self, x: int) -> int:
        return self.sz[self.find(x)]

    def groups(self) -> List[List[int]]:
        roots = defaultdict(list)
        for i in range(len(self.p)):
            roots[self.find(i)].append(i)
        return list(roots.values())

    def roots(self) -> Iterator[int]:
        seen = set()
        for i in range(len(self.p)):
            r = self.find(i)
            if r not in seen:
                seen.add(r)
                yield r

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[tuple[int, int]]) -> "DSU":
        d = cls(n)
        for a, b in edges:
            d.union(a, b)
        return d

class HashDisjointSet:
    def __init__(self):
        self.index = {}          # hash → int index
        self.rev = []            # int → hash
        self.dsu = None

    def _ensure(self, h):
        if h not in self.index:
            self.index[h] = len(self.rev)
            self.rev.append(h)
            if self.dsu:
                # expand existing DSU
                self.dsu.p.append(len(self.rev)-1)
                self.dsu.sz.append(1)
            else:
                self.dsu = DisjointSet(1)
        return self.index[h]

    def union(self, ha, hb):
        ia, ib = self._ensure(ha), self._ensure(hb)
        self.dsu.union(ia, ib)

    def same(self, ha, hb):
        ia, ib = self._ensure(ha), self._ensure(hb)
        return self.dsu.same(ia, ib)

    def groups(self):
        if self.dsu is None:
            return []
        
        groups = self.dsu.groups()
        return [[self.rev[i] for i in g] for g in groups]
