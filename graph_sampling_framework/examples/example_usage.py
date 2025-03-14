import dgl
import json
from graph_sampling.framework import SamplingFramework

# 加载配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

# 创建一个示例DGL图
graph = dgl.graph(([0, 1, 2], [1, 2, 0]))

# 初始化采样框架
framework = SamplingFramework(config)

# 执行采样
result = framework.run(graph)
print(result)
