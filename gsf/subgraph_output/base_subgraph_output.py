# Base class for Subgraph_output
# subgraph_output/base_subgraph_output.py
from abc import ABC, abstractmethod
import dgl

class BaseSubgraphOutput(ABC):
    """
    子图输出模块基类
    """
    def __init__(self, config):
        """
        构造函数

        Args:
            config (dict): 子图输出模块的配置参数
        """
        self.config = config
        self.output_path = config.get("output_path") # 获取输出路径，所有子类都可能用到
        self.output_format = config.get("output_format", "dgl_graph") # 获取默认输出格式

    @abstractmethod
    def output_subgraph(self, subgraph: dgl.DGLGraph):
        """
        抽象方法：输出子图数据

        Args:
            subgraph (dgl.DGLGraph): 要输出的子图，以 DGLGraph 对象形式
        """
        pass