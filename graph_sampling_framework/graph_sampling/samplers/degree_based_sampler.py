# graph_sampling/samplers/degree_based_sampler.py
import dgl
import numpy as np
from .base_sampler import BaseSampler

class DegreeBasedSampler(BaseSampler):
    def sample(self, graph, config):
        num_nodes_to_sample = config["num_nodes"]
        degrees = graph.in_degrees() if graph.is_directed() else graph.degrees()
        prob = degrees / degrees.sum()
        sampled_nodes = np.random.choice(graph.num_nodes(), size=num_nodes_to_sample, replace=False, p=prob)
        subgraph = dgl.node_subgraph(graph, sampled_nodes)
        return subgraph
