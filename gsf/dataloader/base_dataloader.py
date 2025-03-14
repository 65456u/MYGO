# Base class for Dataloader
# dataloader/base_dataloader.py
from abc import ABC, abstractmethod
import dgl

class BaseDataLoader(ABC):
    """
    数据加载模块基类
    """
    def __init__(self, config):
        """
        构造函数

        Args:
            config (dict): 数据加载模块的配置参数
        """
        self.config = config

    @abstractmethod
    def load_graph(self) -> dgl.DGLGraph:
        """
        抽象方法：加载图数据

        Returns:
            dgl.DGLGraph: 加载的图数据，以 DGLGraph 对象形式返回
        """
        pass

    def preprocess_features(self, graph, features):
        """
        预处理节点特征 (可选，子类可重写)

        Args:
            graph (dgl.DGLGraph): 图数据对象
            features (numpy.ndarray): 节点特征矩阵

        Returns:
            numpy.ndarray: 预处理后的节点特征矩阵
        """
        if features is None:
            return None

        if self.config.get("feature_normalize", False):
            # 特征归一化示例 (可以根据需要扩展)
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-6 # 避免除以 0
            features = (features - mean) / std
        return features