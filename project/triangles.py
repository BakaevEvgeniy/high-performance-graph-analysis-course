from typing import List
from pygraphblas import Matrix, Vector
from pygraphblas.types import BOOL, INT64
from pygraphblas.descriptor import R, RC, S


def triangles_for_each_vertex(graph: Matrix) -> List[int]:
    """Counts the number of triangles that include each vertex in the given undirected graph. 

    Args:
        graph (Matrix): undirected graph as an adjacency matrix

    Returns:
        List[int]: a list of integers representing the number of triangles each vertex belongs to
    """
    squared = graph.mxm(graph, cast=INT64, mask=graph, desc=S)
    triangles = squared.reduce_vector()

    return [triangles.get(i, default=0) // 2 for i in range(triangles.size)]


def triangles_cohen(graph: Matrix) -> int:
    """ Counts the number of triangles in an undirected graph using Cohen's algorithm.

    Args:
        graph (Matrix): undirected graph as an adjacency matrix

    Returns:
        int: total triangles in the graph
    """
    result = graph.tril().mxm(graph.triu(), cast=INT64, mask=graph, desc=S)
    return result.reduce() // 2


def triangles_sandia(graph: Matrix) -> int:
    """ Counts the number of triangles in an undirected graph using Sandia algorithm.

    Args:
        graph (Matrix): undirected graph as an adjacency matrix

    Returns:
        int: total triangles in the graph
    """
    tril = graph.tril()
    result = tril.mxm(tril, cast=INT64, mask=tril, desc=S)
    return result.reduce()
