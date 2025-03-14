# Example implementation for Edgelist Loader
# dataloader/edgelist_loader.py
import dgl
import torch as th
import numpy as np
from .base_dataloader import BaseDataLoader # 导入基类

class EdgeListLoader(BaseDataLoader):
    """
    从 Edge List 文件加载图数据的数据加载器
    """
    def __init__(self, config):
        """
        构造函数

        Args:
            config (dict): EdgeListLoader 的配置参数，需要包含:
                data_path (str): Edge List 文件路径
                feature_path (str, optional): 特征文件路径 (可选). Defaults to None.
                feature_dtype (str, optional): 特征数据类型. Defaults to "float32".
                delimiter (str, optional): Edge List 文件分隔符. Defaults to None (空格分隔).
        """
        super().__init__(config)
        self.data_path = config.get("data_path")
        self.feature_path = config.get("feature_path", None)
        self.feature_dtype = config.get("feature_dtype", "float32")
        self.delimiter = config.get("delimiter", None) # 允许配置分隔符，默认为空格

    def load_graph(self) -> dgl.DGLGraph:
        """
        加载 Edge List 文件，并加载节点特征 (如果提供)

        Returns:
            dgl.DGLGraph: 加载的图数据
        """
        src_nodes = []
        dst_nodes = []
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: # 跳过空行
                    continue
                parts = line.split(self.delimiter) if self.delimiter else line.split() # 根据分隔符分割
                if len(parts) != 2:
                    raise ValueError(f"Edge list line must contain two node IDs, but got: {line}")
                try:
                    u, v = map(int, parts)
                except ValueError:
                    raise ValueError(f"Invalid node IDs in line: {line}. Node IDs must be integers.")
                src_nodes.append(u)
                dst_nodes.append(v)

        if not src_nodes: # 检查是否成功读取到边
            raise ValueError(f"No edges found in Edge List file: {self.data_path}")

        src_nodes = th.tensor(src_nodes)
        dst_nodes = th.tensor(dst_nodes)
        graph = dgl.graph((src_nodes, dst_nodes))

        if self.feature_path:
            try:
                features = np.load(self.feature_path).astype(self.feature_dtype)
            except FileNotFoundError:
                raise FileNotFoundError(f"Feature file not found: {self.feature_path}")
            except Exception as e:
                raise ValueError(f"Error loading feature file: {self.feature_path}. {e}")

            if features.shape[0] != graph.num_nodes():
                raise ValueError(f"Number of features ({features.shape[0]}) does not match number of nodes in graph ({graph.num_nodes()}).")

            features = self.preprocess_features(graph, features) # 调用基类的预处理方法
            graph.ndata['feat'] = th.tensor(features)

        return graph