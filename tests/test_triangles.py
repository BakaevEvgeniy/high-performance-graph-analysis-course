import pytest
from project.triangles import (
    triangles_for_each_vertex,
    triangles_cohen,
    triangles_sandia,
)
from project.utils import undirected_graph_to_adj_matrix, read_graph


@pytest.mark.parametrize(
    "graph, expected_ans",
    [
        ("./tests/test_cycled_triangles_graph.dot", [2, 1, 2, 1]),
        ("./tests/test_zero_triangles_graph.dot", [0] * 3),
        ("./tests/test_separated_triangles_graph.dot", [1] * 6),
    ],
)
def test_triangles_for_each_vertex(graph, expected_ans):
    adj = undirected_graph_to_adj_matrix(read_graph(graph))
    assert triangles_for_each_vertex(adj) == expected_ans


@pytest.mark.parametrize(
    "graph, expected_ans",
    [
        ("./tests/test_cycled_triangles_graph.dot", 2),
        ("./tests/test_zero_triangles_graph.dot", 0),
        ("./tests/test_separated_triangles_graph.dot", 2),
    ],
)
def test_triangles_cohen(graph, expected_ans):
    adj = undirected_graph_to_adj_matrix(read_graph(graph))
    assert triangles_cohen(adj) == expected_ans


@pytest.mark.parametrize(
    "graph, expected_ans",
    [
        ("./tests/test_cycled_triangles_graph.dot", 2),
        ("./tests/test_zero_triangles_graph.dot", 0),
        ("./tests/test_separated_triangles_graph.dot", 2),
    ],
)
def test_triangles_sandia(graph, expected_ans):
    adj = undirected_graph_to_adj_matrix(read_graph(graph))
    assert triangles_sandia(adj) == expected_ans
