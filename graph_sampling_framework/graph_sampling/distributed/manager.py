import torch
import torch.distributed as dist

class DistributedManager:
    def __init__(self, num_workers):
        """
        初始化分布式管理器。
        
        参数:
            num_workers: int, 分布式worker数量。
        """
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")
        
        self.num_workers = num_workers
        self.rank = 0  # 默认单机模式
        self.world_size = 1  # 默认单机模式
        
        if num_workers > 1:
            dist.init_process_group(backend="gloo")  # 使用gloo后端，适合CPU分布式
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def run_sampling(self, graph, sampler, config):
        """
        执行分布式采样。
        
        参数:
            graph: dgl.DGLGraph, 输入的图数据。
            sampler: BaseSampler, 采样器实例。
            config: dict, 采样配置参数。
        
        返回:
            采样结果，具体格式取决于采样器和处理器。
        """
        if self.world_size == 1:  # 单机模式
            return sampler.sample(graph, config)

        # 分布式模式：按节点分配采样任务
        total_nodes = graph.number_of_nodes()
        nodes_per_worker = total_nodes // self.world_size
        start_node = self.rank * nodes_per_worker
        end_node = start_node + nodes_per_worker if self.rank < self.world_size - 1 else total_nodes

        # 调整采样配置，按worker分配节点
        local_config = config.copy()
        if "num_nodes" in config:
            local_config["num_nodes"] = config["num_nodes"] // self.world_size
            if self.rank == self.world_size - 1:  # 最后一个worker处理剩余节点
                local_config["num_nodes"] += config["num_nodes"] % self.world_size

        # 执行本地采样
        local_result = sampler.sample(graph, local_config)

        # 收集所有worker的结果
        gathered_results = [None] * self.world_size
        dist.all_gather_object(gathered_results, local_result)

        # 合并结果（仅在rank 0上处理）
        if self.rank == 0:
            merged_result = []
            for result in gathered_results:
                if result:
                    merged_result.extend(result)
            return merged_result
        return None

    def cleanup(self):
        """
        清理分布式环境。
        """
        if self.world_size > 1:
            dist.destroy_process_group()