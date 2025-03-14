import dgl
from .base_sampler import BaseSampler
import numpy as np
import pickle
import os

class RandomNodeSampler(BaseSampler):
    def sample(self, graph, config):
        """
        从图中随机选择指定数量的节点，支持批次采样。
        
        参数:
            graph: dgl.DGLGraph, 输入的图数据。
            config: dict, 包含采样参数，如 {
                "num_nodes": 10,
                "num_batches": 1,
                "allow_overlap": False,
                "save_to_file": False,
                "output_dir": "batches"
            }
        
        返回:
            list of list 或 generator: 每个批次的采样节点ID列表，或无限采样生成器。
        """
        num_nodes = config.get("num_nodes")
        num_batches = config.get("num_batches", 1)
        allow_overlap = config.get("allow_overlap", False)
        save_to_file = config.get("save_to_file", False)
        output_dir = config.get("output_dir", "batches")

        if num_nodes is None or num_nodes <= 0:
            raise ValueError("config must contain 'num_nodes' and it should be positive")
        total_nodes = graph.number_of_nodes()

        # 创建保存目录
        if save_to_file:
            os.makedirs(output_dir, exist_ok=True)

        # 不允许重复，且 num_batches=-1 时，限制最大批次数
        if not allow_overlap and num_batches == -1:
            max_batches = total_nodes // num_nodes
            if max_batches == 0:
                raise ValueError(f"Cannot sample {num_nodes} nodes without overlap from a graph with {total_nodes} nodes")
            num_batches = max_batches

        # 允许重复，且 num_batches=-1 时，返回无限生成器
        if allow_overlap and num_batches == -1:
            def infinite_sampler():
                while True:
                    sampled_nodes = np.random.choice(total_nodes, size=num_nodes, replace=True).tolist()
                    if save_to_file:
                        subgraph = dgl.node_subgraph(graph, sampled_nodes)
                        batch_file = os.path.join(output_dir, f"batch_{np.random.randint(0, 1000000)}.pkl")
                        with open(batch_file, 'wb') as f:
                            pickle.dump(subgraph, f)
                    yield sampled_nodes
            return infinite_sampler()  # 修改为调用函数，返回生成器对象

        # 有限批次采样
        sampled_node_sets = []
        remaining_nodes = np.arange(total_nodes)

        for batch_idx in range(num_batches if num_batches != -1 else total_nodes // num_nodes):
            if not allow_overlap and len(remaining_nodes) < num_nodes:
                break  # 不够节点继续采样，结束

            if allow_overlap:
                sampled_nodes = np.random.choice(total_nodes, size=num_nodes, replace=True)
            else:
                sampled_indices = np.random.choice(len(remaining_nodes), size=num_nodes, replace=False)
                sampled_nodes = remaining_nodes[sampled_indices]
                remaining_nodes = np.delete(remaining_nodes, sampled_indices)

            sampled_nodes = sampled_nodes.tolist()
            if save_to_file:
                subgraph = dgl.node_subgraph(graph, sampled_nodes)
                batch_file = os.path.join(output_dir, f"batch_{batch_idx}.pkl")
                with open(batch_file, 'wb') as f:
                    pickle.dump(subgraph, f)
            sampled_node_sets.append(sampled_nodes)

        return sampled_node_sets