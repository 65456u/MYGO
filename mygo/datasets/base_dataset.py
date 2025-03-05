from abc import ABC, abstractmethod
class BaseDataset(ABC):
    @abstractmethod
    def get_dataloaders(self, batch_size, use_ddp, seed):
        """
        Abstract method to get train, validation, and test dataloaders.
        """
        pass