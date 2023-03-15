from typing import List
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