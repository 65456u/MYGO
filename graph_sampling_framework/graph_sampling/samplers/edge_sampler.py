# graph_sampling/samplers/edge_sampler.py
import dgl
from .base_sampler import BaseSampler

class EdgeSampler(BaseSampler):
    def sample(self, graph, config):
        num_edges = config["num_edges"]
        edges = dgl.sampling.random_edges(graph, num_edges)
        return edges
