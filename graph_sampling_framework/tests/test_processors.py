import dgl
import types
from graph_sampling.processors.dgl_processor import DGLProcessor

def test_dgl_processor_node_list():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    processor = DGLProcessor(output_format="node_list")
    sampled_data = [[0, 2], [1, 3]]
    result = processor.process(sampled_data, graph)
    assert isinstance(result, list), "结果应为列表"
    assert result == [[0, 2], [1, 3]], "节点列表不正确"

def test_dgl_processor_subgraph():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    processor = DGLProcessor(output_format="subgraph")
    sampled_data = [[0, 2], [1, 3]]
    result = processor.process(sampled_data, graph)
    assert isinstance(result, list), "结果应为列表"
    assert len(result) == 2, "应包含2个子图"
    assert result[0].number_of_nodes() == 2, "子图节点数量应为2"

def test_dgl_processor_infinite_generator():
    graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
    processor = DGLProcessor(output_format="subgraph")
    def infinite_generator():
        while True:
            yield [0, 2]
    result = processor.process(infinite_generator(), graph)
    assert isinstance(result, types.GeneratorType), "应返回生成器"
    batch = next(result)
    assert isinstance(batch, dgl.DGLGraph), "生成器应返回DGL子图"
    assert batch.number_of_nodes() == 2, "子图节点数量应为2"