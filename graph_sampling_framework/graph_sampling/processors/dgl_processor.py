import dgl
import types  # 添加导入

class DGLProcessor:
    def __init__(self, output_format="node_list"):
        """
        初始化输出处理器。
        
        参数:
            output_format: str, 输出格式（"node_list" 或 "subgraph"）。
        """
        self.output_format = output_format

    def process(self, sampled_data, graph):
        """
        处理采样结果并转换为指定格式。
        
        参数:
            sampled_data: 采样结果（可能是列表或生成器）。
            graph: dgl.DGLGraph, 原始图数据。
        
        返回:
            处理后的结果，可能是列表或生成器。
        """
        # 优先检查是否是生成器
        if isinstance(sampled_data, types.GeneratorType):
            def processed_generator():
                for nodes in sampled_data:
                    if self.output_format == "node_list":
                        yield nodes
                    elif self.output_format == "subgraph":
                        if not isinstance(nodes, list):
                            raise ValueError("Each batch must be a list of node IDs for subgraph output")
                        yield dgl.node_subgraph(graph, nodes)
                    else:
                        raise ValueError(f"Unsupported output format: {self.output_format}")
            return processed_generator()

        # 如果是列表
        if not isinstance(sampled_data, list):
            raise ValueError("sampled_data must be a list or generator")
        
        if not sampled_data or not isinstance(sampled_data[0], list):
            sampled_data = [sampled_data]
        
        if self.output_format == "node_list":
            return sampled_data
        elif self.output_format == "subgraph":
            return [dgl.node_subgraph(graph, nodes) for nodes in sampled_data]
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")