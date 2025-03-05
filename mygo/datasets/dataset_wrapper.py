from dgl.data import GINDataset, split_dataset
from dgl.dataloading import GraphDataLoader
from mygo.datasets.base_dataset import BaseDataset

class DGLDatasetWrapper(BaseDataset):
    def __init__(self, name="IMDBBINARY", self_loop=False):
        self.dataset = GINDataset(name=name, self_loop=self_loop)

    def get_dataloaders(self, batch_size, use_ddp, seed):
        """Prepare data loaders for training, validation, and test."""
        # Use a 80:10:10 train-val-test split
        train_set, val_set, test_set = split_dataset(
            self.dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
        )
        train_loader = GraphDataLoader(
            train_set, use_ddp=use_ddp, batch_size=batch_size, shuffle=True
        )
        val_loader = GraphDataLoader(val_set, batch_size=batch_size)
        test_loader = GraphDataLoader(test_set, batch_size=batch_size)
        return train_loader, val_loader, test_loader

def load_dataset(name="IMDBBINARY", self_loop=False):
    """Loads the specified DGL dataset."""
    return DGLDatasetWrapper(name=name, self_loop=self_loop)