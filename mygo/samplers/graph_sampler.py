from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader
from .base_sampler import BaseSampler

class GraphDGLSampler(BaseSampler):
    def get_train_dataloader(self, dataset, batch_size, use_ddp, seed):
        train_set, _, _ = split_dataset(
            dataset.dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
        )
        return GraphDataLoader(
            train_set, use_ddp=use_ddp, batch_size=batch_size, shuffle=True
        )

    def get_eval_dataloader(self, dataset, batch_size):
        _, eval_set, _ = split_dataset(
            dataset.dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=0
        )
        return GraphDataLoader(eval_set, batch_size=batch_size)

    def get_test_dataloader(self, dataset, batch_size):
        _, _, test_set = split_dataset(
            dataset.dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=0
        )
        return GraphDataLoader(test_set, batch_size=batch_size)

def create_sampler(): # 简单工厂模式，如果需要更多 Sampler 可以扩展
    return GraphDGLSampler()