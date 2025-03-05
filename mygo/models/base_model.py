from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for the forward pass of the model.
        """
        pass