import pytest
from project.shortest_path import (
    multiple_source_bellman_ford,
    all_pairs_floyd_warshall,
    single_source_bellman_ford,
)
from project.utils import weighted_digraph_to_ajd_matrix, read_graph
import numpy as np


@pytest.mark.parametrize(
    "graph, start_vertices, expected_ans",
    [
        (
            "./tests/test_cycled_triangles_graph.dot",
            [0, 1],
            [(0, [0.0, 1.0, 1.0, 1.0]), (1, [np.inf, 0.0, 1.0, 2.0])],
        ),
        (
            "./tests/test_zero_triangles_graph.dot",
            [0, 1],
            [
                (0, [0.0, 10.0, -5.0]),
                (1, [np.inf, 0.0, np.inf]),
            ],
        ),
        (
            "./tests/test_separated_triangles_graph.dot",
            [0, 1, 2, 3, 4, 5],
            [
                (0, [0.0, 1.0, 1.0, np.inf, np.inf, np.inf]),
                (1, [np.inf, 0.0, np.inf, np.inf, np.inf, np.inf]),
                (2, [np.inf, 1.0, 0.0, np.inf, np.inf, np.inf]),
                (3, [np.inf, np.inf, np.inf, 0.0, 1.0, 1.0]),
                (4, [np.inf, np.inf, np.inf, np.inf, 0.0, 1.0]),
                (5, [np.inf, np.inf, np.inf, np.inf, np.inf, 0.0]),
            ],
        ),
        (
            "./tests/test_with_negative_edge_cycle.dot",
            [2, 1],
            [(2, [5.0, 10.0, 0.0]), (1, [-5.0, 0.0, -10.0])],
        ),
    ],
)
def test_multiple_source_bellman_ford(graph, start_vertices, expected_ans):
    adj = weighted_digraph_to_ajd_matrix(read_graph(graph))
    assert multiple_source_bellman_ford(adj, start_vertices) == expected_ans

    for i, v in enumerate(start_vertices):
        assert single_source_bellman_ford(adj, v) == expected_ans[i][1]


@pytest.mark.parametrize(
    "graph, expected_ans",
    [
        (
            "./tests/test_cycled_triangles_graph.dot",
            [
                (0, [0.0, 1.0, 1.0, 1.0]),
                (1, [np.inf, 0.0, 1.0, 2.0]),
                (2, [np.inf, np.inf, 0.0, 1.0]),
                (3, [np.inf, np.inf, np.inf, 0.0]),
            ],
        ),
        (
            "./tests/test_zero_triangles_graph.dot",
            [
                (0, [0.0, 10.0, -5.0]),
                (1, [np.inf, 0.0, np.inf]),
                (2, [np.inf, np.inf, 0.0]),
            ],
        ),
        (
            "./tests/test_separated_triangles_graph.dot",
            [
                (0, [0.0, 1.0, 1.0, np.inf, np.inf, np.inf]),
                (1, [np.inf, 0.0, np.inf, np.inf, np.inf, np.inf]),
                (2, [np.inf, 1.0, 0.0, np.inf, np.inf, np.inf]),
                (3, [np.inf, np.inf, np.inf, 0.0, 1.0, 1.0]),
                (4, [np.inf, np.inf, np.inf, np.inf, 0.0, 1.0]),
                (5, [np.inf, np.inf, np.inf, np.inf, np.inf, 0.0]),
            ],
        ),
        (
            "./tests/test_with_negative_edge_cycle.dot",
            [(0, [0.0, 5.0, -5.0]), (1, [-5.0, 0.0, -10.0]), (2, [5.0, 10.0, 0.0])],
        ),
    ],
)
def test_all_pairs_floyd_warshall(graph, expected_ans):
    adj = weighted_digraph_to_ajd_matrix(read_graph(graph))
    assert all_pairs_floyd_warshall(adj) == expected_ans
