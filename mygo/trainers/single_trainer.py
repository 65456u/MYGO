# File: ./mygo/trainers/single_trainer.py
from loguru import logger
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..models import BaseModel
from ..samplers import BaseSampler
from ..utils.training_args import TrainingArguments
from .base_trainer import BaseTrainer

class SingleTrainer(BaseTrainer):
    """Trainer for single GPU or CPU training."""

    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        args: TrainingArguments,
        sampler: BaseSampler = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        self.device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.sampler = sampler

        self._setup_model()
        if self.is_local_main_process():
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def _setup_model(self):
        """Set up the model for single GPU/CPU training."""
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def is_local_main_process(self):
        """For single GPU, it's always the main process."""
        return True

    def train_step(self, batch):
        """Perform a single training step."""
        bg, labels = batch
        bg = bg.to(self.device)
        labels = labels.to(self.device)
        feats = bg.ndata.pop("attr")
        outputs = self.model(bg, feats)
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader."""
        self.model.eval()
        total = 0
        total_correct = 0
        for bg, labels in dataloader:
            bg = bg.to(self.device)
            labels = labels.to(self.device)
            feats = bg.ndata.pop("attr")
            with torch.no_grad():
                pred = self.model(bg, feats)
            _, pred = torch.max(pred, 1)
            total += len(labels)
            total_correct += (pred == labels).sum().cpu().item()
        return 1.0 * total_correct / total

    def train(self):
        """Train the model."""
        logger.info(f"Starting single GPU training...")
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            if self.sampler and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            if self.is_local_main_process():
                logger.info(f"Epoch {epoch}: Training started.")

            total_loss = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss, _ = self.train_step(batch)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.args.logging_steps == 0 and self.is_local_main_process():
                    global_step = epoch * len(self.train_dataloader) + batch_idx
                    self.writer.add_scalar('Loss/train', loss.item(), global_step)

            avg_loss = total_loss / len(self.train_dataloader)
            val_acc = self.evaluate(self.val_dataloader)

            if self.is_local_main_process():
                logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}, Validation Accuracy = {val_acc:.4f}")
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Val acc: {val_acc:.4f}")
                self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
                self.writer.add_scalar('Accuracy/validation', val_acc, epoch)

    def run(self):
        """Run the full training and evaluation process."""
        self.train()
        test_acc = self.evaluate(self.test_dataloader)
        if self.is_local_main_process():
            logger.info(f"Test Accuracy = {test_acc:.4f}")
            print(f"Test acc: {test_acc:.4f}")
            self.writer.add_scalar('Accuracy/test', test_acc, 0)
            self.writer.close()