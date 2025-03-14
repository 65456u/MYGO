# subgraph_output/__init__.py
# 使 subgraph_output 目录成为 Python 包

from .base_subgraph_output import BaseSubgraphOutput
from .dgl_graph_output import DGLGraphOutput

__all__ = [
    'BaseSubgraphOutput',
    'DGLGraphOutput',
]