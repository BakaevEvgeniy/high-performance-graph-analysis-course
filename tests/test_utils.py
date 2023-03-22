from project.utils import read_graph, graph_to_ajd_matrix


def test_read_graph():
    graph = read_graph("./tests/test_graph.dot")

    adj = graph_to_ajd_matrix(graph)

    sources = [0, 0, 1, 2, 3, 4, 5]
    targets = [1, 3, 2, 0, 4, 5, 0]

    assert adj.to_lists()[0] == sources
    assert adj.to_lists()[1] == targets
    assert adj.to_lists()[2] == [True] * 7
