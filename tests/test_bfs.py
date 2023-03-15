import pytest
from pygraphblas import Matrix
from pygraphblas.types import BOOL, INT64
from project.bfs import bfs


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 0, 1, 1, 3, 5, 5, ],
            [1, 5, 2, 3, 5, 6, 4, ],
            [True] * 7,
            7,
            0,
            [0, 1, 2, 2, 2, 1, 2],
        ),
        ([0], [0], [False], 1, 0, [0]),
        ([0, 1, 1], [1, 1, 2], [True, True, False], 3, 0, [0, 1, -1]),
        (
            [0, 0, 1, 1, 3, 3, 5, 5, ],
            [1, 5, 2, 3, 2, 5, 6, 4, ],
            [True] * 8,
            7,
            3,
            [-1, -1, 1, 0, 2, 1, 2],
        ),
        (
            [0, 0, 1, 1, 3, 3, 5, 5, ],
            [1, 5, 2, 3, 2, 5, 6, 4, ],
            [True] * 8,
            7,
            1,
            [-1, 0, 1, 1, 3, 2, 3],
        ),
    ],
)
def test_bfs(I, J, V, size, start_vertex, expected_ans):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert bfs(graph, start_vertex) == expected_ans


