import dgl
from graph_sampling.framework import SamplingFramework

def test_framework_with_random_node_sampler_node_list():
    # 创建一个简单的图（5个节点，环形结构）
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))

    # 配置：使用 RandomNodeSampler，采样3个节点，输出格式为 node_list，单机模式
    config = {
        "sampler": "RandomNodeSampler",
        "sampler_params": {"num_nodes": 3},
        "output_format": "node_list",
        "distributed": False,
        "num_workers": 1
    }

    # 初始化并运行框架
    framework = SamplingFramework(config)
    result = framework.run(graph)
    framework.cleanup()

    # 验证结果
    assert isinstance(result, list), "结果应为节点ID列表"
    assert len(result) == 3, "采样节点数量应为3"
    assert all(0 <= node < 5 for node in result), "采样节点ID应在范围内"

def test_framework_with_random_node_sampler_subgraph():
    # 创建一个简单的图（5个节点，环形结构）
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))

    # 配置：使用 RandomNodeSampler，采样3个节点，输出格式为 subgraph，单机模式
    config = {
        "sampler": "RandomNodeSampler",
        "sampler_params": {"num_nodes": 3},
        "output_format": "subgraph",
        "distributed": False,
        "num_workers": 1
    }

    # 初始化并运行框架
    framework = SamplingFramework(config)
    result = framework.run(graph)
    framework.cleanup()

    # 验证结果
    assert isinstance(result, dgl.DGLGraph), "结果应为DGL子图"
    assert result.number_of_nodes() == 3, "子图节点数量应为3"