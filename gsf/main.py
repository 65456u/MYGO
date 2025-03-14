# main.py
from config.config_manager import ConfigManager

if __name__ == '__main__':
    config_path = "config/config.yaml" # 确保路径正确
    config_manager = ConfigManager(config_path)
    sampled_subgraph, eval_report = config_manager.run_sampling_pipeline()