from networkx import MultiDiGraph, nx_agraph
from pygraphblas import Matrix
from pygraphblas.types import BOOL


def read_graph(path: str) -> MultiDiGraph:
    return nx_agraph.read_dot(path)


def graph_to_ajd_matrix(graph: MultiDiGraph) -> Matrix:
    adj_matrix = Matrix.sparse(BOOL, graph.number_of_nodes(), graph.number_of_nodes())
    for source, target in graph.edges():
        adj_matrix[int(source), int(target)] = True
    return adj_matrix


def undirected_graph_to_adj_matrix(graph: MultiDiGraph) -> Matrix:
    adj_matrix = Matrix.sparse(BOOL, graph.number_of_nodes(), graph.number_of_nodes())
    for source, target in graph.edges():
        adj_matrix[int(source), int(target)] = True
        adj_matrix[int(target), int(source)] = True
    return adj_matrix
