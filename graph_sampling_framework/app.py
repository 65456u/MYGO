import gradio as gr
import pickle
import dgl
import os
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image  # 添加PIL导入
from graph_sampling.framework import SamplingFramework

# 加载图数据
def load_graph(file, graph_format):
    if file is None:
        return "请上传图数据文件", None
    
    try:
        with open(file.name, 'rb') as f:
            if graph_format == "dgl":
                graph = pickle.load(f)
                if not isinstance(graph, dgl.DGLGraph):
                    return "上传的文件不是有效的DGL图", None
            elif graph_format == "pyg":
                return "PyG格式暂未实现", None
            elif graph_format == "networkx":
                graph = pickle.load(f)
                if not isinstance(graph, nx.Graph):
                    return "上传的文件不是有效的NetworkX图", None
                graph = dgl.from_networkx(graph)
            elif graph_format == "csv":
                return "CSV格式暂未实现", None
            elif graph_format == "json":
                return "JSON格式暂未实现", None
            else:
                return "未知图格式", None
        
        return f"图加载成功：节点数 = {graph.number_of_nodes()}, 边数 = {graph.number_of_edges()}", graph
    except Exception as e:
        return f"加载图数据失败: {str(e)}", None

# 执行采样并返回迭代器结果
def run_sampling(graph_state, sampler_type, sampler_params, output_format, batch_index, allow_overlap, save_to_file, output_dir):
    if graph_state is None:
        return "请先加载有效的图数据", None, None
    
    print(f"采样参数: sampler_type={sampler_type}, sampler_params={sampler_params}, output_format={output_format}, "
          f"allow_overlap={allow_overlap}, save_to_file={save_to_file}, output_dir={output_dir}")
    
    try:
        params = eval(sampler_params) if sampler_params else {}
        if not isinstance(params, dict):
            return "采样参数格式错误，应为字典（如 {'num_nodes': 10, 'num_batches': 5}）", None, None
        
        params["allow_overlap"] = allow_overlap
        params["save_to_file"] = save_to_file
        params["output_dir"] = output_dir
        
        config = {
            "sampler": sampler_type,
            "sampler_params": params,
            "output_format": output_format
        }
        framework = SamplingFramework(config)
        result = framework.run(graph_state)
        
        if callable(result):  # 生成器（无限采样）
            iterator = result
            batches = []
            for i, batch in enumerate(iterator):
                if i == batch_index:
                    if output_format == "subgraph":
                        return (f"批次 {batch_index} 子图: 节点数 = {batch.number_of_nodes()}, 边数 = {batch.number_of_edges()}",
                                batch, iterator)
                    else:
                        return f"批次 {batch_index} 节点列表: {batch}", batch, iterator
                batches.append(batch)
                if i >= 100:
                    break
            return f"批次 {batch_index} 超出范围（已采样 {len(batches)} 批次）", None, iterator
        else:  # 有限批次采样
            if batch_index >= len(result):
                return f"批次 {batch_index} 超出范围（共 {len(result)} 批次）", None, None
            batch_result = result[batch_index]
            if output_format == "subgraph":
                return (f"批次 {batch_index} 子图: 节点数 = {batch_result.number_of_nodes()}, 边数 = {batch_result.number_of_edges()}",
                        batch_result, None)
            else:
                return str(batch_result), batch_result, None
    except Exception as e:
        print(f"采样出错: {str(e)}")
        return f"采样出错: {str(e)}", None, None

# 可视化图
def visualize_graph(graph_state):
    if graph_state is None:
        return None
    
    try:
        # 将DGL图转换为NetworkX图以便可视化
        if isinstance(graph_state, dgl.DGLGraph):
            nx_graph = graph_state.to_networkx()
        else:
            return None
        
        # 使用Matplotlib绘制图
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=10, arrows=True)
        plt.title("Graph Visualization")
        
        # 将图转换为PIL图像
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)  # 转换为PIL图像对象
        plt.close()
        return img
    except Exception as e:
        print(f"可视化失败: {str(e)}")
        return None

# Gradio界面
with gr.Blocks(title="图采样框架") as demo:
    gr.Markdown("# 图采样框架前端界面")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传图数据文件")
            graph_format = gr.Dropdown(
                choices=["dgl", "pyg", "networkx", "csv", "json"],
                label="图格式",
                value="dgl"
            )
            load_btn = gr.Button("加载图")
            graph_info = gr.Textbox(label="图信息", interactive=False)
            graph_state = gr.State()
            
        with gr.Column():
            sampler_type = gr.Dropdown(
                choices=["RandomNodeSampler"],
                label="采样器类型",
                value="RandomNodeSampler"
            )
            sampler_params = gr.Textbox(
                label="采样参数",
                placeholder='{"num_nodes": 10, "num_batches": 5} 或 {"num_nodes": 10, "num_batches": -1}',
                value='{"num_nodes": 10, "num_batches": 5}'
            )
            output_format = gr.Dropdown(
                choices=["node_list", "subgraph"],
                label="输出格式",
                value="subgraph"
            )
            allow_overlap = gr.Checkbox(label="允许批次间节点重复", value=False)
            save_to_file = gr.Checkbox(label="保存批次到文件", value=False)
            output_dir = gr.Textbox(label="保存目录", placeholder="batches", value="batches")
            batch_index = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                label="批次索引",
                value=0
            )
            sample_btn = gr.Button("执行采样")
            sample_output = gr.Textbox(label="采样结果", interactive=False)
            batch_state = gr.State()
            iterator_state = gr.State()

    with gr.Row():
        visualize_btn = gr.Button("可视化")
        visualize_output = gr.Image(label="可视化结果", type="pil")

    load_btn.click(
        fn=load_graph,
        inputs=[file_input, graph_format],
        outputs=[graph_info, graph_state]
    )
    sample_btn.click(
        fn=run_sampling,
        inputs=[graph_state, sampler_type, sampler_params, output_format, batch_index, allow_overlap, save_to_file, output_dir],
        outputs=[sample_output, batch_state, iterator_state]
    )
    visualize_btn.click(
        fn=visualize_graph,
        inputs=batch_state,
        outputs=visualize_output
    )

# 启动界面
demo.launch(debug=True)  # 添加 share=True 创建公共链接