import os

# 项目根目录
project_root = "graph_sampling_framework"

# 创建目录
dirs = [
    "graph_sampling",
    "graph_sampling/loaders",
    "graph_sampling/samplers",
    "graph_sampling/processors",
    "graph_sampling/distributed",
    "tests",
    "examples"
]
for d in dirs:
    os.makedirs(os.path.join(project_root, d), exist_ok=True)

# 文件内容字典
files_content = {
    "graph_sampling/__init__.py": "# graph_sampling/__init__.py\nfrom .framework import SamplingFramework\n",
    "graph_sampling/loaders/__init__.py": "# graph_sampling/loaders/__init__.py\nfrom .dgl_loader import DGLLoader\n",
    "graph_sampling/loaders/dgl_loader.py": "# graph_sampling/loaders/dgl_loader.py\nimport dgl\n\nclass DGLLoader:\n    def load(self, graph_data):\n        if isinstance(graph_data, dgl.DGLGraph):\n            return graph_data\n        else:\n            raise ValueError(\"Input must be a DGLGraph\")\n",
    "graph_sampling/samplers/__init__.py": "# graph_sampling/samplers/__init__.py\nfrom .base_sampler import BaseSampler\nfrom .random_walk_sampler import RandomWalkSampler\nfrom .neighbor_sampler import NeighborSampler\nfrom .node_sampler import RandomNodeSampler\nfrom .edge_sampler import EdgeSampler\nfrom .subgraph_extractor import SubgraphExtractor\nfrom .layerwise_sampler import LayerwiseSampler\nfrom .degree_based_sampler import DegreeBasedSampler\nfrom .pagerank_based_sampler import PageRankBasedSampler\nfrom .community_based_sampler import CommunityBasedSampler\n",
    "graph_sampling/samplers/base_sampler.py": "# graph_sampling/samplers/base_sampler.py\nfrom abc import ABC, abstractmethod\n\nclass BaseSampler(ABC):\n    @abstractmethod\n    def sample(self, graph, config):\n        pass\n",
    "graph_sampling/samplers/random_walk_sampler.py": "# graph_sampling/samplers/random_walk_sampler.py\nimport dgl\nfrom .base_sampler import BaseSampler\n\nclass RandomWalkSampler(BaseSampler):\n    def sample(self, graph, config):\n        start_nodes = config[\"start_nodes\"]\n        walk_length = config[\"walk_length\"]\n        walks = dgl.sampling.random_walk(graph, start_nodes, length=walk_length)\n        return walks\n",
    "graph_sampling/samplers/neighbor_sampler.py": "# graph_sampling/samplers/neighbor_sampler.py\nimport dgl\nfrom .base_sampler import BaseSampler\n\nclass NeighborSampler(BaseSampler):\n    def sample(self, graph, config):\n        nodes = config[\"nodes\"]\n        num_neighbors = config[\"num_neighbors\"]\n        subgraph = dgl.sampling.sample_neighbors(graph, nodes, num_neighbors)\n        return subgraph\n",
    "graph_sampling/samplers/node_sampler.py": "# graph_sampling/samplers/node_sampler.py\nimport dgl\nfrom .base_sampler import BaseSampler\n\nclass RandomNodeSampler(BaseSampler):\n    def sample(self, graph, config):\n        num_nodes = config[\"num_nodes\"]\n        nodes = dgl.sampling.random_nodes(graph, num_nodes)\n        return nodes\n",
    "graph_sampling/samplers/edge_sampler.py": "# graph_sampling/samplers/edge_sampler.py\nimport dgl\nfrom .base_sampler import BaseSampler\n\nclass EdgeSampler(BaseSampler):\n    def sample(self, graph, config):\n        num_edges = config[\"num_edges\"]\n        edges = dgl.sampling.random_edges(graph, num_edges)\n        return edges\n",
    "graph_sampling/samplers/subgraph_extractor.py": "# graph_sampling/samplers/subgraph_extractor.py\nimport dgl\nfrom .base_sampler import BaseSampler\n\nclass SubgraphExtractor(BaseSampler):\n    def sample(self, graph, config):\n        nodes = config[\"nodes\"]\n        subgraph = dgl.node_subgraph(graph, nodes)\n        return subgraph\n",
    "graph_sampling/samplers/layerwise_sampler.py": "# graph_sampling/samplers/layerwise_sampler.py\nfrom .base_sampler import BaseSampler\n\nclass LayerwiseSampler(BaseSampler):\n    def sample(self, graph, config):\n        # 待实现\n        pass\n",
    "graph_sampling/samplers/degree_based_sampler.py": "# graph_sampling/samplers/degree_based_sampler.py\nimport dgl\nimport numpy as np\nfrom .base_sampler import BaseSampler\n\nclass DegreeBasedSampler(BaseSampler):\n    def sample(self, graph, config):\n        num_nodes_to_sample = config[\"num_nodes\"]\n        degrees = graph.in_degrees() if graph.is_directed() else graph.degrees()\n        prob = degrees / degrees.sum()\n        sampled_nodes = np.random.choice(graph.num_nodes(), size=num_nodes_to_sample, replace=False, p=prob)\n        subgraph = dgl.node_subgraph(graph, sampled_nodes)\n        return subgraph\n",
    "graph_sampling/samplers/pagerank_based_sampler.py": "# graph_sampling/samplers/pagerank_based_sampler.py\nimport torch\nimport numpy as np\nfrom .base_sampler import BaseSampler\n\nclass PageRankBasedSampler(BaseSampler):\n    def sample(self, graph, config):\n        num_nodes_to_sample = config[\"num_nodes\"]\n        num_iterations = config.get(\"num_iterations\", 10)\n        alpha = config.get(\"alpha\", 0.85)\n        PR = self.compute_pagerank(graph, num_iterations, alpha)\n        prob = PR / PR.sum()\n        sampled_nodes = np.random.choice(graph.num_nodes(), size=num_nodes_to_sample, replace=False, p=prob.cpu().numpy())\n        subgraph = dgl.node_subgraph(graph, sampled_nodes)\n        return subgraph\n\n    def compute_pagerank(self, graph, num_iterations, alpha):\n        num_nodes = graph.num_nodes()\n        A = graph.adjacency_matrix(transpose=True, scipy_fmt='coo')\n        A = A.todense()\n        A = torch.from_numpy(A).to(dtype=torch.float32)\n        PR = torch.full((num_nodes,), 1 / num_nodes)\n        for _ in range(num_iterations):\n            PR = alpha * (torch.mm(PR.unsqueeze(0), A).squeeze(0)) + (1 - alpha) / num_nodes\n        return PR\n",
    "graph_sampling/samplers/community_based_sampler.py": "# graph_sampling/samplers/community_based_sampler.py\nfrom .base_sampler import BaseSampler\n\nclass CommunityBasedSampler(BaseSampler):\n    def sample(self, graph, config):\n        # 待实现\n        pass\n",
    "graph_sampling/processors/__init__.py": "# graph_sampling/processors/__init__.py\nfrom .dgl_processor import DGLProcessor\n",
    "graph_sampling/processors/dgl_processor.py": "# graph_sampling/processors/dgl_processor.py\nclass DGLProcessor:\n    def __init__(self, format=\"dgl\"):\n        self.format = format\n\n    def process(self, sampled_data):\n        return sampled_data\n",
    "graph_sampling/distributed/__init__.py": "# graph_sampling/distributed/__init__.py\nfrom .manager import DistributedManager\n",
    "graph_sampling/distributed/manager.py": "# graph_sampling/distributed/manager.py\nclass DistributedManager:\n    def __init__(self, num_workers):\n        self.num_workers = num_workers\n\n    def run_sampling(self, graph, sampler, config):\n        # 简化为单机实现\n        return sampler.sample(graph, config)\n",
    "graph_sampling/framework.py": "# graph_sampling/framework.py\nfrom .loaders.dgl_loader import DGLLoader\nfrom .samplers import *\nfrom .processors.dgl_processor import DGLProcessor\nfrom .distributed.manager import DistributedManager\n\nclass SamplingFramework:\n    def __init__(self, config):\n        self.loader = DGLLoader()\n        self.sampler = globals()[config[\"sampler\"]]()\n        self.processor = DGLProcessor(config[\"output\"])\n        self.distributed = config.get(\"distributed\", False)\n        self.manager = DistributedManager(config.get(\"num_workers\", 1)) if self.distributed else None\n\n    def run(self, graph_data):\n        graph = self.loader.load(graph_data)\n        if self.distributed:\n            sampled_data = self.manager.run_sampling(graph, self.sampler, config)\n        else:\n            sampled_data = self.sampler.sample(graph, config)\n        return self.processor.process(sampled_data)\n",
    "tests/__init__.py": "",
    "tests/test_loaders.py": "# tests/test_loaders.py\nimport dgl\nfrom graph_sampling.loaders.dgl_loader import DGLLoader\n\ndef test_dgl_loader():\n    loader = DGLLoader()\n    graph = dgl.graph(([0, 1], [1, 2]))\n    loaded_graph = loader.load(graph)\n    assert loaded_graph == graph\n",
    "tests/test_samplers.py": "# tests/test_samplers.py\nimport dgl\nfrom graph_sampling.samplers.random_walk_sampler import RandomWalkSampler\n\ndef test_random_walk_sampler():\n    graph = dgl.graph(([0, 1, 2], [1, 2, 0]))\n    config = {\"start_nodes\": [0], \"walk_length\": 2}\n    sampler = RandomWalkSampler()\n    walks = sampler.sample(graph, config)\n    assert walks.shape[1] == 3  # walk_length + 1\n",
    "tests/test_processors.py": "# tests/test_processors.py\nfrom graph_sampling.processors.dgl_processor import DGLProcessor\n\ndef test_dgl_processor():\n    processor = DGLProcessor(format=\"dgl\")\n    data = \"sample_data\"\n    processed_data = processor.process(data)\n    assert processed_data == data\n",
    "tests/test_framework.py": "# tests/test_framework.py\nimport dgl\nimport json\nfrom graph_sampling.framework import SamplingFramework\n\ndef test_framework():\n    config = {\n        \"sampler\": \"RandomWalkSampler\",\n        \"sampler_params\": {\"start_nodes\": [0], \"walk_length\": 2},\n        \"output\": \"dgl\",\n        \"distributed\": False\n    }\n    graph = dgl.graph(([0, 1, 2], [1, 2, 0]))\n    framework = SamplingFramework(config)\n    result = framework.run(graph)\n    assert result is not None\n",
    "examples/config.json": "{\n    \"sampler\": \"RandomWalkSampler\",\n    \"sampler_params\": {\n        \"start_nodes\": [0, 1],\n        \"walk_length\": 5\n    },\n    \"output\": \"dgl\",\n    \"distributed\": false\n}\n",
    "examples/example_usage.py": "import dgl\nimport json\nfrom graph_sampling.framework import SamplingFramework\n\n# 加载配置文件\nwith open('config.json', 'r') as f:\n    config = json.load(f)\n\n# 创建一个示例DGL图\ngraph = dgl.graph(([0, 1, 2], [1, 2, 0]))\n\n# 初始化采样框架\nframework = SamplingFramework(config)\n\n# 执行采样\nresult = framework.run(graph)\nprint(result)\n",
    "requirements.txt": "dgl\ntorch\nnumpy\n",
    "README.md": "# Graph Sampling Framework\n\n这是一个图采样框架，支持多种采样方法和灵活的配置。\n\n## 安装\n\npip install -r requirements.txt\n\n## 使用\n\n见examples/example_usage.py\n"
}

# 创建文件并写入内容
for file_path, content in files_content.items():
    with open(os.path.join(project_root, file_path), "w") as f:
        f.write(content)

print("项目文件和目录已成功创建！")