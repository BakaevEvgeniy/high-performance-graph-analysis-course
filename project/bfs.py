from typing import List, Tuple
from pygraphblas import Matrix, Vector
from pygraphblas.types import BOOL, INT64
from pygraphblas.descriptor import R, RC


def bfs(graph: Matrix, start_vertex: int) -> List[int]:
    n = graph.ncols
    q = Vector.sparse(BOOL, n)
    used = Vector.sparse(BOOL, n)
    ans = Vector.dense(INT64, n, fill=-1)
    ans[start_vertex] = 0

    used[start_vertex] = True
    q[start_vertex] = True
    step = 1
    prev_nnz = -1
    while used.nvals != prev_nnz:
        prev_nnz = used.nvals
        q.vxm(graph, mask=used, desc=RC, out=q)
        used.eadd(q, BOOL.lor_land, desc=R, out=used)
        ans.assign_scalar(step, mask=q)
        step += 1

    return list(ans.vals)


def multi_source_bfs(
    graph: Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    n = graph.ncols
    m = len(start_vertices)
    q = Matrix.sparse(BOOL, nrows=m, ncols=n)
    used = Matrix.sparse(BOOL, nrows=m, ncols=n)
    ans = Matrix.dense(INT64, nrows=m, ncols=n, fill=-1)

    for i, j in enumerate(start_vertices):
        q.assign_scalar(True, i, j)
        used.assign_scalar(True, i, j)
        ans.assign_scalar(0, i, j)

    step = 1
    prev_nnz = -1
    while used.nvals != prev_nnz:
        prev_nnz = used.nvals
        q.mxm(graph, mask=used, out=q, desc=RC)
        used.eadd(q, BOOL.lor_land, out=used, desc=R)
        ans.assign_scalar(step, mask=q)
        step += 1

    return [(vertex, list(ans[i].vals)) for i, vertex in enumerate(start_vertices)]
