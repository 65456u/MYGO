# Example implementation for Dgl Graph Output
# subgraph_output/dgl_graph_output.py
import dgl
from .base_subgraph_output import BaseSubgraphOutput # 导入基类

class DGLGraphOutput(BaseSubgraphOutput):
    """
    输出 DGLGraph 对象到文件的子图输出模块
    """
    def __init__(self, config):
        """
        构造函数

        Args:
            config (dict): DGLGraphOutput 的配置参数，需要包含:
                output_path (str): 输出文件路径
        """
        super().__init__(config)

    def output_subgraph(self, subgraph: dgl.DGLGraph):
        """
        保存 DGLGraph 对象到文件

        Args:
            subgraph (dgl.DGLGraph): 要输出的子图
        """
        try:
            dgl.save_graphs(self.output_path, [subgraph])
            print(f"子图已保存为 DGLGraph 格式到: {self.output_path}")
        except Exception as e:
            print(f"保存 DGLGraph 子图到文件失败: {self.output_path}. 错误信息: {e}")