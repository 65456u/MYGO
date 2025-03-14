# graph_sampling/samplers/random_walk_sampler.py
import dgl
from .base_sampler import BaseSampler

class RandomWalkSampler(BaseSampler):
    def sample(self, graph, config):
        start_nodes = config["start_nodes"]
        walk_length = config["walk_length"]
        walks = dgl.sampling.random_walk(graph, start_nodes, length=walk_length)
        return walks
