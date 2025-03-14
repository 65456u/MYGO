from abc import ABC, abstractmethod

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, graph, config):
        """
        采样方法，子类必须实现。
        
        参数:
            graph: dgl.DGLGraph, 输入的图数据。
            config: dict, 采样配置参数。
        
        返回:
            采样结果，具体类型由子类定义。
        """
        pass