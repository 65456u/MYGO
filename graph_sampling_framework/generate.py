import dgl
import torch
import pickle

def generate_test_graph(num_nodes=1024  , output_file="test_graph_full.pkl"):
    """
    生成一个全连接图并保存为 .pkl 文件。

    参数:
        num_nodes (int): 节点数，默认为65536。
        output_file (str): 输出文件路径，默认为"test_graph.pkl"。

    注意:
        全连接图的边数为 num_nodes * (num_nodes - 1)，
        节点数过大可能会导致内存不足。
    """
    # 生成节点编号
    nodes = torch.arange(num_nodes)

    # 生成所有可能的边（包括自环），然后过滤掉自环
    src = nodes.repeat(num_nodes)
    dst = nodes.unsqueeze(1).repeat(1, num_nodes).flatten()
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # 创建DGL图
    graph = dgl.graph((src, dst))

    # 验证图的节点数和边数
    assert graph.number_of_nodes() == num_nodes, f"预期节点数为 {num_nodes}，实际为 {graph.number_of_nodes()}"
    expected_edges = num_nodes * (num_nodes - 1)
    assert graph.number_of_edges() == expected_edges, f"预期边数为 {expected_edges}，实际为 {graph.number_of_edges()}"

    # 保存为 .pkl 文件
    with open(output_file, 'wb') as f:
        pickle.dump(graph, f)

    print(f"成功生成全连接图：节点数 = {num_nodes}，边数 = {expected_edges}，已保存到 {output_file}")

if __name__ == "__main__":
    # 生成全连接图
    generate_test_graph()
