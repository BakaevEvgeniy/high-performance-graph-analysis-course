from typing import List
from pygraphblas import Matrix, Vector
from pygraphblas.types import BOOL, INT64
from pygraphblas.descriptor import R, RC, S


def triangles_for_each_vertex(graph: Matrix) -> List[int]:
    squared = graph.mxm(graph, cast=INT64, mask=graph, desc=S)
    triangles = squared.reduce_vector()

    return [triangles.get(i, default=0) // 2 for i in range(triangles.size)]


def triangles_cohen(graph: Matrix) -> int:
    result = graph.tril().mxm(graph.triu(), cast=INT64, mask=graph, desc=S)
    return result.reduce() // 2


def triangles_sandia(graph: Matrix) -> int:
    tril = graph.tril()
    result = tril.mxm(tril, cast=INT64, mask=tril, desc=S)
    return result.reduce()
