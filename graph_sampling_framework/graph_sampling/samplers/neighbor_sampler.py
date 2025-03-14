# graph_sampling/samplers/neighbor_sampler.py
import dgl
from .base_sampler import BaseSampler

class NeighborSampler(BaseSampler):
    def sample(self, graph, config):
        nodes = config["nodes"]
        num_neighbors = config["num_neighbors"]
        subgraph = dgl.sampling.sample_neighbors(graph, nodes, num_neighbors)
        return subgraph
