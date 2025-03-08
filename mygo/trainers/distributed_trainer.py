from loguru import logger
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..models import BaseModel     # From model package
from ..samplers import BaseSampler # From sampler package
from ..utils.training_args import TrainingArguments # From utils package
from .base_trainer import BaseTrainer

class DistributedTrainer(BaseTrainer): # Inherit from BaseTrainer
    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        args: TrainingArguments, # Type hint for TrainingArguments
        rank,
        world_size,
        sampler: BaseSampler = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.sampler = sampler

        self._setup_distributed()
        self._setup_model()
        if self.is_local_main_process():
            self.writer = SummaryWriter(log_dir=args.log_dir)

    def _setup_distributed(self):
        if self.args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            logger.info(f"Rank {self.rank}: Using GPU {self.device}")
        else:
            logger.info(f"Rank {self.rank}: Using CPU")

        # if self.world_size > 1:
        #     dist.init_process_group(
        #         backend=self.args.ddp_backend,
        #         init_method="tcp://127.0.0.1:12345",
        #         world_size=self.world_size,
        #         rank=self.rank,
        #     )
        #     logger.info(f"Rank {self.rank}: Process group initialized.")

    def _setup_model(self):
        self.model.to(self.device)
        if self.world_size > 1:
            if self.args.device == "cpu":
                self.model = DistributedDataParallel(self.model)
                logger.info(f"Rank {self.rank}: Model wrapped with DDP on CPU.")
            else:
                self.model = DistributedDataParallel(
                    self.model, device_ids=[self.device], output_device=self.device
                )
                logger.info(f"Rank {self.rank}: Model wrapped with DDP on GPU {self.device}.")

    def is_local_main_process(self):
        return self.rank == 0

    def train_step(self, batch):
        bg, labels = batch
        bg = bg.to(self.device)
        labels = labels.to(self.device)
        feats = bg.ndata.pop("attr")
        outputs = self.model(bg, feats)
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def evaluate(self, dataloader):
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
        logger.info(f"Rank {self.rank}: Starting training...")
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

            # --- Metric Aggregation and Logging ---
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            acc_tensor = torch.tensor([val_acc], device=self.device)

            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
            gathered_accs = [torch.zeros_like(acc_tensor) for _ in range(self.world_size)]

            if self.world_size > 1:
                dist.all_gather(gathered_losses, loss_tensor)
                dist.all_gather(gathered_accs, acc_tensor)

                global_avg_loss = torch.mean(torch.stack(gathered_losses)).item()
                global_avg_acc = torch.mean(torch.stack(gathered_accs)).item()
            else:
                global_avg_loss = avg_loss
                global_avg_acc = val_acc

            if self.is_local_main_process():
                logger.info(f"Epoch {epoch}: Average Loss = {global_avg_loss:.4f}, Validation Accuracy = {global_avg_acc:.4f}")
                print(f"Epoch {epoch}, Loss: {global_avg_loss:.4f}, Val acc: {global_avg_acc:.4f}")
                self.writer.add_scalar('Loss/epoch', global_avg_loss, epoch)
                self.writer.add_scalar('Accuracy/validation', global_avg_acc, epoch)

    def run(self):
        self.train()
        test_acc = self.evaluate(self.test_dataloader)
        if self.is_local_main_process():
            logger.info(f"Test Accuracy = {test_acc:.4f}")
            print(f"Test acc: {test_acc:.4f}")
            self.writer.add_scalar('Accuracy/test', test_acc, 0)
            self.writer.close()

        if self.world_size > 1:
            dist.destroy_process_group()
            logger.info(f"Rank {self.rank}: Process group destroyed.")