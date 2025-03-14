import dgl
from graph_sampling.loaders.dgl_loader import DGLLoader

def test_dgl_loader():
    # 创建一个简单的DGL图（2条边：0->1, 1->2）
    graph = dgl.graph(([0, 1], [1, 2]))
    
    # 初始化加载器
    loader = DGLLoader()
    
    # 加载图
    loaded_graph = loader.load(graph)
    
    # 验证加载后的图与原始图相同
    assert loaded_graph == graph, "加载后的图与原始图不一致"
    assert isinstance(loaded_graph, dgl.DGLGraph), "加载后的图不是DGLGraph类型"

def test_dgl_loader_invalid_input():
    # 初始化加载器
    loader = DGLLoader()
    
    # 测试无效输入（例如字符串）
    try:
        loader.load("invalid_input")
        assert False, "应该抛出ValueError"
    except ValueError:
        pass  # 预期抛出异常，测试通过