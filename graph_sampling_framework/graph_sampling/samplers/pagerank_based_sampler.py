# graph_sampling/samplers/pagerank_based_sampler.py
import torch
import numpy as np
from .base_sampler import BaseSampler

class PageRankBasedSampler(BaseSampler):
    def sample(self, graph, config):
        num_nodes_to_sample = config["num_nodes"]
        num_iterations = config.get("num_iterations", 10)
        alpha = config.get("alpha", 0.85)
        PR = self.compute_pagerank(graph, num_iterations, alpha)
        prob = PR / PR.sum()
        sampled_nodes = np.random.choice(graph.num_nodes(), size=num_nodes_to_sample, replace=False, p=prob.cpu().numpy())
        subgraph = dgl.node_subgraph(graph, sampled_nodes)
        return subgraph

    def compute_pagerank(self, graph, num_iterations, alpha):
        num_nodes = graph.num_nodes()
        A = graph.adjacency_matrix(transpose=True, scipy_fmt='coo')
        A = A.todense()
        A = torch.from_numpy(A).to(dtype=torch.float32)
        PR = torch.full((num_nodes,), 1 / num_nodes)
        for _ in range(num_iterations):
            PR = alpha * (torch.mm(PR.unsqueeze(0), A).squeeze(0)) + (1 - alpha) / num_nodes
        return PR
