import dgl
import os
import pickle
import types
from graph_sampling.samplers.node_sampler import RandomNodeSampler

def test_random_node_sampler_no_overlap():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    config = {"num_nodes": 2, "num_batches": 2, "allow_overlap": False}
    sampler = RandomNodeSampler()
    result = sampler.sample(graph, config)
    
    assert len(result) == 2, "应返回2个批次"
    assert len(result[0]) == 2 and len(result[1]) == 2, "每个批次应包含2个节点"
    assert len(set(result[0]).intersection(set(result[1]))) == 0, "批次间节点不应重复"

def test_random_node_sampler_with_overlap():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    config = {"num_nodes": 2, "num_batches": 3, "allow_overlap": True}
    sampler = RandomNodeSampler()
    result = sampler.sample(graph, config)
    
    assert len(result) == 3, "应返回3个批次"
    assert len(result[0]) == 2 and len(result[1]) == 2 and len(result[2]) == 2, "每个批次应包含2个节点"

def test_random_node_sampler_infinite_sampling():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    config = {"num_nodes": 2, "num_batches": -1, "allow_overlap": True}
    sampler = RandomNodeSampler()
    result = sampler.sample(graph, config)
    
    assert isinstance(result, types.GeneratorType), "应返回生成器"
    batches = [next(result) for _ in range(5)]
    assert len(batches) == 5, "应生成5个批次"
    assert all(len(batch) == 2 for batch in batches), "每个批次应包含2个节点"

def test_random_node_sampler_save_to_file():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    config = {"num_nodes": 2, "num_batches": 2, "allow_overlap": False, "save_to_file": True, "output_dir": "test_batches"}
    sampler = RandomNodeSampler()
    result = sampler.sample(graph, config)
    
    assert len(result) == 2, "应返回2个批次"
    assert os.path.exists("test_batches/batch_0.pkl"), "批次文件应存在"
    with open("test_batches/batch_0.pkl", 'rb') as f:
        subgraph = pickle.load(f)
        assert isinstance(subgraph, dgl.DGLGraph), "保存的文件应为DGL图"
    # 清理测试目录
    for file in os.listdir("test_batches"):
        os.remove(os.path.join("test_batches", file))
    os.rmdir("test_batches")