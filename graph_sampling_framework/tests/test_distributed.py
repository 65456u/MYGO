import dgl
from graph_sampling.distributed.manager import DistributedManager
from graph_sampling.samplers.node_sampler import RandomNodeSampler

def test_distributed_manager_single_worker():
    # 创建一个简单的图（5个节点，环形结构）
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    
    # 初始化采样器
    sampler = RandomNodeSampler()
    
    # 配置：采样3个节点
    config = {"num_nodes": 3}
    
    # 初始化分布式管理器（单机模式）
    manager = DistributedManager(num_workers=1)
    
    # 执行采样
    result = manager.run_sampling(graph, sampler, config)
    
    # 验证结果
    assert isinstance(result, list), "结果应为节点ID列表"
    assert len(result) == 3, "采样节点数量应为3"
    assert all(0 <= node < 5 for node in result), "采样节点ID应在范围内"
    
    # 清理
    manager.cleanup()

# 分布式模式测试需要在多进程环境中运行，此处提供示例代码
"""
import torch.multiprocessing as mp

def run_worker(rank, world_size, graph, sampler, config):
    manager = DistributedManager(num_workers=world_size)
    result = manager.run_sampling(graph, sampler, config)
    print(f"Rank {rank} result: {result}")
    manager.cleanup()

def test_distributed_manager_multi_worker():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    sampler = RandomNodeSampler()
    config = {"num_nodes": 3}
    world_size = 2
    mp.spawn(run_worker, args=(world_size, graph, sampler, config), nprocs=world_size, join=True)
"""