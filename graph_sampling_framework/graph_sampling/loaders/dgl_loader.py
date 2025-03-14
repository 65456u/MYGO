# graph_sampling/loaders/dgl_loader.py
import dgl

class DGLLoader:
    def load(self, graph_data):
        if isinstance(graph_data, dgl.DGLGraph):
            return graph_data
        else:
            raise ValueError("Input must be a DGLGraph")
