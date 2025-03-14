# Base class for Sampler
# sampler/base_sampler.py
from abc import ABC, abstractmethod
import dgl

class BaseSampler(ABC):
    """
    采样算法模块基类
    """
    def __init__(self, config, graph: dgl.DGLGraph):
        """
        构造函数

        Args:
            config (dict): 采样算法模块的配置参数
            graph (dgl.DGLGraph): 原始图数据，由数据加载模块加载
        """
        self.config = config
        self.graph = graph

    @abstractmethod
    def sample(self) -> dgl.DGLGraph:
        """
        抽象方法：执行采样算法

        Returns:
            dgl.DGLGraph: 采样得到的子图，以 DGLGraph 对象形式返回
        """
        pass