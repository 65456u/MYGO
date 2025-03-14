# config/config_manager.py
import yaml
import time
from dataloader import EdgeListLoader # 导入数据加载模块 (示例)
from sampler import RandomNodeSampler   # 导入采样算法模块 (示例)
from subgraph_output import DGLGraphOutput # 导入子图输出模块 (示例)
from evaluator import TopologyEvaluator   # 导入评估模块 (示例)

class ConfigManager:
    """
    配置管理器，负责加载配置、实例化模块和控制流程
    """
    def __init__(self, config_path: str):
        """
        构造函数

        Args:
            config_path (str): 配置文件路径 (YAML 格式)
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.data_loader = None
        self.sampler = None
        self.subgraph_output = None
        self.evaluator = None

    def load_config(self) -> dict:
        """
        加载 YAML 配置文件

        Returns:
            dict: 配置字典
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件解析错误: {self.config_path}. 错误信息: {e}")

    def initialize_modules(self):
        """
        根据配置文件实例化各个模块
        """
        config = self.config

        # 数据加载模块
        data_loader_config = config.get("data_loader", {})
        data_loader_module_name = data_loader_config.get("module_name")
        if data_loader_module_name == "EdgeListLoader":
            self.data_loader = EdgeListLoader(data_loader_config)
        else:
            raise ValueError(f"Unknown DataLoader module: {data_loader_module_name}")

        # 采样算法模块
        sampler_config = config.get("sampler", {})
        sampler_module_name = sampler_config.get("module_name")
        if sampler_module_name == "RandomNodeSampler":
            self.sampler = RandomNodeSampler(sampler_config, None) # Graph 对象在 run_sampling_pipeline 中传入
        else:
            raise ValueError(f"Unknown Sampler module: {sampler_module_name}")

        # 子图输出模块
        subgraph_output_config = config.get("subgraph_output", {})
        subgraph_output_module_name = subgraph_output_config.get("module_name")
        if subgraph_output_module_name == "DGLGraphOutput":
            self.subgraph_output = DGLGraphOutput(subgraph_output_config)
        else:
            raise ValueError(f"Unknown SubgraphOutput module: {subgraph_output_module_name}")

        # 性能评估模块
        evaluator_config = config.get("evaluator", {})
        evaluator_module_name = evaluator_config.get("module_name")
        if evaluator_module_name == "TopologyEvaluator":
            self.evaluator = TopologyEvaluator(evaluator_config)
        elif evaluator_module_name is None: # 允许不配置 evaluator 模块
            self.evaluator = None
            print("Warning: Evaluator module is not configured in config.yaml.")
        else:
            raise ValueError(f"Unknown Evaluator module: {evaluator_module_name}")

    def run_sampling_pipeline(self):
        """
        运行整个采样流程

        Returns:
            tuple: 包含采样子图 (dgl.DGLGraph) 和评估报告 (dict, 如果配置了 Evaluator 模块) 的元组
                   如果未配置 Evaluator 模块，则评估报告为 None
        """
        self.initialize_modules() # 初始化模块

        # 1. 数据加载
        print("加载图数据...")
        original_graph = self.data_loader.load_graph()
        self.sampler.graph = original_graph # 将加载的图数据传递给 Sampler

        # 2. 采样
        print("执行采样算法...")
        start_time = time.time()
        sampled_subgraph = self.sampler.sample()
        end_time = time.time()

        # 3. 子图输出
        print("输出子图数据...")
        self.subgraph_output.output_subgraph(sampled_subgraph)

        # 4. 性能评估 (可选)
        evaluation_report = None
        if self.evaluator:
            print("进行性能评估...")
            evaluation_report = self.evaluator.evaluate(original_graph, sampled_subgraph)
            evaluation_report.update(self.evaluator.evaluate_sampling_time(start_time, end_time)) # 添加采样时间
            self.evaluator.print_evaluation_report(evaluation_report) # 打印评估报告
        else:
            print("跳过性能评估.")

        print("采样流程完成!")
        return sampled_subgraph, evaluation_report