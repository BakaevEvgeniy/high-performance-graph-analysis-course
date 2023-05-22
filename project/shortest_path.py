from typing import List, Tuple
from pygraphblas import Matrix, Vector
from pygraphblas.types import BOOL, INT64, FP64
from pygraphblas.descriptor import R, RC, S
import numpy as np


def multiple_source_bellman_ford(
    graph: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[float]]]:
    """Finds the shortest paths in an oriented graph of several given vertices.

    Args:
        graph (Matrix): directed graph as an adjacency matrix (FP64)
        start_vertices (List[int]): list of starting vertices

    Raises:
        ValueError: if there is a negative cycle in the graph

    Returns:
        List[Tuple[int, List[float]]]: List of tuples: a vertex, and list where for each vertex the distance to it from the specified one is given. If the vertex is not reachable, the value of the corresponding cell is np.inf.
    """
    n = graph.nrows

    if n == 0 or len(start_vertices) == 0:
        return []

    graph = graph.eadd(Matrix.identity(FP64, n, 0.0), FP64.MIN)
    dist = Matrix.sparse(FP64, len(start_vertices), n)

    for i, start in enumerate(start_vertices):
        dist[i, start] = 0

    for iter_num in range(n):
        old = dist
        dist = dist.mxm(graph, FP64.MIN_PLUS)
        if old.iseq(dist):
            print(f"Finish on {iter_num}/{n} iteration (finish/max).")
            return [
                (
                    start,
                    [dist.get(i, col, default=np.inf) for col in range(n)],
                )
                for i, start in enumerate(start_vertices)
            ]

    raise ValueError("Negative weight cycle detected")


def single_source_bellman_ford(graph: Matrix, start: int) -> List[float]:
    """Finds shortest paths from one initial vertex to all other vertices in a given weighted graph.

    Args:
        graph (Matrix): directed graph as an adjacency matrix (FP64)
        start (int): start vertex

    Returns:
        List[float]: a list of distances from the initial vertex for each vertex in the graph. If the current vertex is unreachable from the initial vertex then its shortest distance is equal to np.inf.
    """
    return multiple_source_bellman_ford(graph, [start])[0][1]


def all_pairs_floyd_warshall(graph: Matrix) -> List[Tuple[int, List[float]]]:
    """Find the shortest paths in an oriented graph for all pairs of vertices

    Args:
        graph (Matrix): directed graph as an adjacency matrix (FP64)

    Raises:
        ValueError: if there is a negative cycle in the graph

    Returns:
        List[Tuple[int, List[float]]]: List of tuples: a vertex, and list where for each vertex the distance to it from current. If the vertex is not reachable, the value of the corresponding cell is np.inf.
    """
    n = graph.nrows
    dist = graph.dup()

    for k in range(n):
        step = dist.extract_matrix(col_index=k).mxm(
            dist.extract_matrix(row_index=k), semiring=FP64.MIN_PLUS
        )
        dist.eadd(step, add_op=FP64.MIN, out=dist)

    for k in range(n):
        step = dist.extract_matrix(col_index=k).mxm(
            dist.extract_matrix(row_index=k), semiring=FP64.MIN_PLUS
        )
        if dist.isne(dist.eadd(step, add_op=FP64.MIN)):
            raise ValueError("Negative weight cycle detected")

    return [(i, [dist.get(i, j, default=np.inf) for j in range(n)]) for i in range(n)]
