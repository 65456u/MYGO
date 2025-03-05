from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    @abstractmethod
    def train(self):
        """Abstract method to train the model."""
        pass

    @abstractmethod
    def evaluate(self, dataloader):
        """Abstract method to evaluate the model."""
        pass

    @abstractmethod
    def run(self):
        """Abstract method to run the full training and evaluation process."""
        pass
