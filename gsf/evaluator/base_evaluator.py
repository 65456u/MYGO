# Base class for Evaluator
# evaluator/base_evaluator.py
from abc import ABC, abstractmethod
import dgl
import time

class BaseEvaluator(ABC):
    """
    性能评估模块基类
    """
    def __init__(self, config):
        """
        构造函数

        Args:
            config (dict): 性能评估模块的配置参数
        """
        self.config = config
        self.metrics = config.get("metrics", []) # 获取要评估的指标列表

    @abstractmethod
    def evaluate(self, original_graph: dgl.DGLGraph, sampled_graph: dgl.DGLGraph) -> dict:
        """
        抽象方法：执行性能评估

        Args:
            original_graph (dgl.DGLGraph): 原始图数据
            sampled_graph (dgl.DGLGraph): 采样子图数据

        Returns:
            dict: 评估报告，包含各种性能指标及其值 (字典形式)
        """
        pass

    def evaluate_sampling_time(self, start_time: float, end_time: float) -> dict:
        """
        评估采样时间 (通用方法)

        Args:
            start_time (float): 采样开始时间 (秒)
            end_time (float): 采样结束时间 (秒)

        Returns:
            dict: 包含采样时间的评估结果，例如: {'sampling_time': 0.123}
        """
        sampling_time = end_time - start_time
        return {"sampling_time": sampling_time}

    def print_evaluation_report(self, evaluation_report: dict):
        """
        打印评估报告到控制台 (通用方法)

        Args:
            evaluation_report (dict): 评估报告字典
        """
        print("性能评估报告:")
        for metric, value in evaluation_report.items():
            print(f"  {metric}: {value}")