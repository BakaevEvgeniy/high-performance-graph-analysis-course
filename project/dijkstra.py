import heapq
import itertools

import boltons.queueutils
import networkx as nx


def dijkstra_sssp(graph: nx.Graph, start):
    distances = {v: float("inf") for v in graph.nodes}
    distances[start] = 0

    q = [(0, start)]

    while q:
        current_distance, v = heapq.heappop(q)
        if current_distance > distances[v]:
            continue

        for neighbor in graph.neighbors(v):
            weight = graph[v][neighbor].get("weight", 1)
            distance = distances[v] + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(q, (distance, neighbor))

    return distances


class DynamicSSSP:
    def __init__(self, graph: nx.DiGraph, start):
        self.graph = graph
        self.start = start
        self.distances = dijkstra_sssp(graph, start)
        self.modified_vertices = set()

    def insert_edge(self, u, v):
        self.graph.add_edge(u, v, weight=1.0)
        self.modified_vertices.add(v)

        if u not in self.distances:
            self.distances[u] = float("inf")
        if v not in self.distances:
            self.distances[v] = float("inf")

    def delete_edge(self, u, v):
        self.graph.remove_edge(u, v)
        self.modified_vertices.add(v)

    def query_distances(self):
        if len(self.modified_vertices) > 0:
            self.update_distances()
            self.modified_vertices = set()
        return self.distances

    def calc_rhs(self, u):
        if u == self.start:
            return 0

        return min(
            (
                self.distances[v] + self.graph[v][u].get("weight", 1.0)
                for v in self.graph.predecessors(u)
            ),
            default=float("inf"),
        )

    def update_distances(self):
        # http://www.ccpo.odu.edu/~klinck/Reprints/PDF/ramalingamJAlgo1996.pdf Fig. 3.

        hpq = boltons.queueutils.HeapPriorityQueue(priority_key=lambda x: x)
        rhs = {}
        for u in self.modified_vertices:
            rhs[u] = self.calc_rhs(u)
            if rhs[u] != self.distances[u]:
                key = min(rhs[u], self.distances[u])
                hpq.add(u, key)

        while hpq:
            u = hpq.pop()
            if rhs[u] < self.distances[u]:
                self.distances[u] = rhs[u]
                for v in self.graph.successors(u):
                    rhs[v] = self.calc_rhs(v)
                    if rhs[v] != self.distances[v]:
                        hpq.add(v, min(rhs[v], self.distances[v]))
                    else:
                        if v in hpq._entry_map:
                            hpq.remove(v)
            else:
                self.distances[u] = float("inf")
                for v in itertools.chain(self.graph.successors(u), [u]):
                    rhs[v] = self.calc_rhs(v)
                    if rhs[v] != self.distances[v]:
                        hpq.add(v, min(rhs[v], self.distances[v]))
                    else:
                        if v in hpq._entry_map:
                            hpq.remove(v)
