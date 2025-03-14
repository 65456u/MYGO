from .loaders.dgl_loader import DGLLoader
from .samplers import BaseSampler, RandomNodeSampler
from .processors.dgl_processor import DGLProcessor
from .distributed.manager import DistributedManager

class SamplingFramework:
    def __init__(self, config):
        """
        初始化采样框架。
        
        参数:
            config: dict, 包含配置参数，例如采样器类型、采样参数和输出格式。
        """
        self.loader = DGLLoader()
        self.sampler = self._get_sampler(config["sampler"])
        self.processor = DGLProcessor(config.get("output_format", "node_list"))
        self.distributed = config.get("distributed", False)
        self.manager = DistributedManager(config.get("num_workers", 1)) if self.distributed else None
        self.config = config

    def _get_sampler(self, sampler_name):
        """
        根据配置获取采样器实例。
        
        参数:
            sampler_name: str, 采样器类名（如 "RandomNodeSampler"）。
        
        返回:
            BaseSampler: 采样器实例。
        """
        sampler_classes = {
            "RandomNodeSampler": RandomNodeSampler
        }
        
        if sampler_name not in sampler_classes:
            raise ValueError(f"未找到采样器 {sampler_name}")
        
        sampler_class = sampler_classes[sampler_name]
        if not issubclass(sampler_class, BaseSampler):
            raise ValueError(f"{sampler_name} 不是 BaseSampler 的子类")
        return sampler_class()

    def run(self, graph_data):
        """
        执行采样流程。
        
        参数:
            graph_data: 输入的图数据。
        
        返回:
            处理后的采样结果（可能是迭代器）。
        """
        graph = self.loader.load(graph_data)
        if self.distributed:
            sampled_data = self.manager.run_sampling(graph, self.sampler, self.config.get("sampler_params", {}))
        else:
            sampled_data = self.sampler.sample(graph, self.config.get("sampler_params", {}))
        return self.processor.process(sampled_data, graph) if sampled_data else None

    def cleanup(self):
        """
        清理框架资源。
        """
        if self.distributed and self.manager:
            self.manager.cleanup()