# Example implementation for Random Node Sampler
# sampler/random_node_sampler.py
import dgl
import torch as th
from .base_sampler import BaseSampler # 导入基类

class RandomNodeSampler(BaseSampler):
    """
    随机节点采样算法
    """
    def __init__(self, config, graph: dgl.DGLGraph):
        """
        构造函数

        Args:
            config (dict): RandomNodeSampler 的配置参数，需要包含:
                num_sampled_nodes (int, optional): 采样的节点数量. Defaults to 100.
        """
        super().__init__(config, graph)
        self.num_sampled_nodes = config.get("num_sampled_nodes", 100)

    def sample(self) -> dgl.DGLGraph:
        """
        随机选择指定数量的节点及其一阶邻居，构建子图

        Returns:
            dgl.DGLGraph: 采样得到的子图
        """
        num_nodes = self.graph.num_nodes()
        if num_nodes <= 0:
            print("Warning: Graph has no nodes. Returning an empty graph.")
            return dgl.graph(([], [])) # 返回空图

        num_sample = min(self.num_sampled_nodes, num_nodes) # 确保采样数量不超过节点总数
        sampled_node_ids = th.randperm(num_nodes)[:num_sample]

        try:
            subgraph = dgl.node_subgraph(self.graph, sampled_node_ids)
        except Exception as e:
            print(f"Error during subgraph sampling: {e}. Returning an empty graph.")
            return dgl.graph(([], [])) # 采样失败时返回空图

        return subgraph