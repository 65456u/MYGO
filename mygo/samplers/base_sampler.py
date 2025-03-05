from abc import ABC, abstractmethod


class BaseSampler(ABC):
    @abstractmethod
    def get_train_dataloader(self, dataset, batch_size, use_ddp, seed):
        """
        Abstract method to get the training dataloader.
        """
        pass

    @abstractmethod
    def get_eval_dataloader(self, dataset, batch_size):
        """
        Abstract method to get the evaluation dataloader (validation/test).
        """
        pass

    @abstractmethod
    def get_test_dataloader(self, dataset, batch_size):
        """
        Abstract method to get the test dataloader.
        """
        pass