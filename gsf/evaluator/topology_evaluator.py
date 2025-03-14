# Example implementation for Topology Evaluator
# evaluator/topology_evaluator.py
import dgl
import time
import numpy as np
from scipy.stats import wasserstein_distance
from .base_evaluator import BaseEvaluator # 导入基类

class TopologyEvaluator(BaseEvaluator):
    """
    评估拓扑保持能力的性能评估模块
    """
    def __init__(self, config):
        """
        构造函数

        Args:
            config (dict): TopologyEvaluator 的配置参数，可以包含要评估的指标列表 (metrics)
        """
        super().__init__(config)

    def evaluate(self, original_graph: dgl.DGLGraph, sampled_graph: dgl.DGLGraph) -> dict:
        """
        执行性能评估，根据配置的指标计算并返回评估报告

        Args:
            original_graph (dgl.DGLGraph): 原始图数据
            sampled_graph (dgl.DGLGraph): 采样子图数据

        Returns:
            dict: 评估报告，包含性能指标及其值
        """
        report = {}
        start_time = time.time() # 记录评估开始时间 (这里仅作为示例，实际采样时间应该在采样模块中记录)
        end_time = start_time # 实际场景中需要记录采样结束时间

        if "sampling_time" in self.metrics:
            report.update(self.evaluate_sampling_time(start_time, end_time)) # 使用基类的通用方法

        if "degree_distribution_similarity" in self.metrics:
            similarity = self.evaluate_degree_distribution_similarity(original_graph, sampled_graph)
            report.update({"degree_distribution_similarity": similarity})

        return report

    def evaluate_degree_distribution_similarity(self, original_graph: dgl.DGLGraph, sampled_graph: dgl.DGLGraph) -> float:
        """
        评估度分布相似性 (使用 Wasserstein 距离)

        Args:
            original_graph (dgl.DGLGraph): 原始图数据
            sampled_graph (dgl.DGLGraph): 采样子图数据

        Returns:
            float: 度分布相似性得分 (例如，基于 Wasserstein 距离转换的相似度)
        """
        original_degrees = original_graph.in_degrees().numpy()
        sampled_degrees = sampled_graph.in_degrees().numpy()
        # 计算 Wasserstein 距离作为度分布差异的度量，距离越小越相似
        distance = wasserstein_distance(original_degrees, sampled_degrees)
        # 将 Wasserstein 距离转换为相似度分数 (距离越小，相似度越高)
        similarity_score = 1.0 / (1.0 + distance) # 简单示例，可以根据需要调整转换方式
        return similarity_score