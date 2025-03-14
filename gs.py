import os

def create_directory(dir_path):
    """创建目录，如果不存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def create_file(file_path, content=""):
    """创建文件，如果不存在，并写入初始内容"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created file: {file_path}")
    else:
        print(f"File already exists: {file_path}")

def create_init_file(dir_path):
    """在指定目录下创建 __init__.py 文件"""
    init_file_path = os.path.join(dir_path, "__init__.py")
    create_file(init_file_path)

def setup_project():
    """初始化项目结构"""
    project_root = "gnn_sampling_framework"
    module_dirs = ["config", "dataloader", "sampler", "subgraph_output", "evaluator"]
    base_class_files = {
        "dataloader": "base_dataloader.py",
        "sampler": "base_sampler.py",
        "subgraph_output": "base_subgraph_output.py",
        "evaluator": "base_evaluator.py"
    }
    example_impl_files = {
        "dataloader": "edgelist_loader.py",
        "sampler": "random_node_sampler.py",
        "subgraph_output": "dgl_graph_output.py",
        "evaluator": "topology_evaluator.py"
    }

    # 创建根目录
    create_directory(project_root)

    # 创建模块目录和 __init__.py 文件
    for module_dir in module_dirs:
        module_path = os.path.join(project_root, module_dir)
        create_directory(module_path)
        create_init_file(module_path)

    # 创建基类文件
    for module, base_class_file in base_class_files.items():
        module_path = os.path.join(project_root, module)
        base_class_path = os.path.join(module_path, base_class_file)
        create_file(base_class_path, content="# Base class for {}\n".format(module.capitalize())) # 添加简单注释

    # 创建示例实现文件
    for module, example_file in example_impl_files.items():
        module_path = os.path.join(project_root, module)
        example_path = os.path.join(module_path, example_file)
        create_file(example_path, content="# Example implementation for {}\n".format(example_file.replace('_', ' ').replace('.py', '').title())) # 添加简单注释

    # 创建 config.yaml 文件 (示例内容)
    config_content = """
framework_name: "GNN Sampling System"
version: "1.0"

data_loader:
  module_name: "EdgeListLoader"
  data_path: "data/cora.edgelist" # 示例路径，需要修改
  feature_path: "data/cora.features.npy" # 示例路径，需要修改
  data_format: "edgelist"
  feature_normalize: true
  feature_dtype: "float32"

sampler:
  module_name: "RandomNodeSampler"
  algorithm_name: "random_node"
  num_sampled_nodes: 100

subgraph_output:
  module_name: "DGLGraphOutput"
  output_format: "dgl_graph"
  output_path: "sampled_subgraph.dgl" # 示例路径，可以修改

evaluator:
  module_name: "TopologyEvaluator"
  metrics: ["sampling_time", "degree_distribution_similarity"]
"""
    create_file(os.path.join(project_root, "config", "config.yaml"), config_content)

    # 创建 main.py 文件 (示例内容)
    main_content = """
from config.config_manager import ConfigManager

if __name__ == '__main__':
    config_path = "config/config.yaml"
    config_manager = ConfigManager(config_path)
    sampled_subgraph, eval_report = config_manager.run_sampling_pipeline()
"""
    create_file(os.path.join(project_root, "main.py"), main_content)

    # 创建 README.md 文件 (可选，可以留空或添加基本项目描述)
    create_file(os.path.join(project_root, "README.md"), content="# GNN Sampling Framework\n\nThis is a framework for Graph Neural Network sampling...")

    print("\nProject setup completed!")
    print(f"Project root directory: {project_root}")

if __name__ == "__main__":
    setup_project()