# graph_sampling/samplers/subgraph_extractor.py
import dgl
from .base_sampler import BaseSampler

class SubgraphExtractor(BaseSampler):
    def sample(self, graph, config):
        nodes = config["nodes"]
        subgraph = dgl.node_subgraph(graph, nodes)
        return subgraph
