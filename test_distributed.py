import os
from loguru import logger

os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dgl.data import GINDataset, split_dataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GINConv, SumPooling
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

def init_process_group(world_size, rank):
    """Initialize the distributed process group."""
    dist.init_process_group(
        backend="nccl",  # Use 'nccl' for multiple GPUs
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )
    logger.info(f"Rank {rank}: Process group initialized.")

def get_dataloaders(dataset, seed, batch_size=512):
    """Prepare data loaders for training, validation, and test."""
    # Use a 80:10:10 train-val-test split
    train_set, val_set, test_set = split_dataset(
        dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
    )
    train_loader = GraphDataLoader(
        train_set, use_ddp=True, batch_size=batch_size, shuffle=True
    )
    val_loader = GraphDataLoader(val_set, batch_size=batch_size)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size)
    logger.info("Data loaders created.")
    return train_loader, val_loader, test_loader

class GIN(nn.Module):
    """Simplified Graph Isomorphism Network."""

    def __init__(self, input_size=1, num_classes=2):
        super().__init__()  # Changed super(GIN, nn.Module).__init__() to super().__init__()

        self.conv1 = GINConv(
            nn.Linear(input_size, num_classes), aggregator_type="sum"
        )
        self.conv2 = GINConv(
            nn.Linear(num_classes, num_classes), aggregator_type="sum"
        )
        self.pool = SumPooling()

    def forward(self, g, feats):
        feats = self.conv1(g, feats)
        feats = F.relu(feats)
        feats = self.conv2(g, feats)
        return self.pool(g, feats)

def init_model(seed, device):
    """Initialize the GIN model and wrap it with DistributedDataParallel."""
    torch.manual_seed(seed)
    model = GIN().to(device)
    if device.type == "cpu":
        model = DistributedDataParallel(model)
        logger.info(f"Rank {device.index if device.type != 'cpu' else 0}: Model initialized on CPU and wrapped with DDP.")
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
        logger.info(f"Rank {device.index}: Model initialized on GPU {device.index} and wrapped with DDP.")
    return model

def evaluate(model, dataloader, device):
    """Evaluate the model on the given dataloader."""
    model.eval()
    total = 0
    total_correct = 0
    for bg, labels in dataloader:
        bg = bg.to(device)
        labels = labels.to(device)
        # Get input node features
        feats = bg.ndata.pop("attr")
        with torch.no_grad():
            pred = model(bg, feats)
        _, pred = torch.max(pred, 1)
        total += len(labels)
        total_correct += (pred == labels).sum().cpu().item()
    return 1.0 * total_correct / total

def run(rank, world_size, dataset, seed=0):
    """Main function for each process."""
    init_process_group(world_size, rank)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        logger.info(f"Rank {rank}: Using GPU {device}")
    else:
        device = torch.device("cpu")
        logger.info(f"Rank {rank}: Using CPU")

    if rank == 0:
        writer = SummaryWriter(log_dir='./runs') # Initialize TensorBoard writer for rank 0, specify log_dir

    model = init_model(seed, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, seed)
    for epoch in range(500):
        model.train()
        # Ensure different random ordering for each epoch
        train_loader.set_epoch(epoch)
        if (rank == 0):
            logger.info(f"Rank {rank}, Epoch {epoch}: Training started.")

        total_loss = 0
        for batch_idx, (bg, labels) in enumerate(train_loader):
            bg = bg.to(device)
            labels = labels.to(device)
            feats = bg.ndata.pop("attr")
            pred = model(bg, feats)
            loss = criterion(pred, labels)
            total_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                # logger.info(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
                if rank == 0: # Log training loss to TensorBoard from rank 0
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(model, val_loader, device)

        # --- Metric Aggregation and Logging ---
        loss_tensor = torch.tensor([avg_loss], device=device)
        acc_tensor = torch.tensor([val_acc], device=device)

        gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        gathered_accs = [torch.zeros_like(acc_tensor) for _ in range(world_size)]

        dist.all_gather(gathered_losses, loss_tensor)
        dist.all_gather(gathered_accs, acc_tensor)

        # Calculate average loss and accuracy across all ranks
        global_avg_loss = torch.mean(torch.stack(gathered_losses)).item()
        global_avg_acc = torch.mean(torch.stack(gathered_accs)).item()

        if rank == 0: # Log only from rank 0
            logger.info(f"Epoch {epoch}: Average Loss = {global_avg_loss:.4f}, Validation Accuracy = {global_avg_acc:.4f}")
            print(f"Epoch {epoch}, Loss: {global_avg_loss:.4f}, Val acc: {global_avg_acc:.4f}")
            writer.add_scalar('Loss/epoch', global_avg_loss, epoch) # Log epoch average loss to TensorBoard
            writer.add_scalar('Accuracy/validation', global_avg_acc, epoch) # Log validation accuracy to TensorBoard

    test_acc = evaluate(model, test_loader, device)
    if rank == 0: # Log only from rank 0 for test accuracy as well
        logger.info(f"Test Accuracy = {test_acc:.4f}")
        print(f"Test acc: {test_acc:.4f}")
        writer.add_scalar('Accuracy/test', test_acc, 0) # Log test accuracy to TensorBoard

    if rank == 0:
        writer.close() # Close TensorBoard writer after logging

    dist.destroy_process_group()
    logger.info(f"Rank {rank}: Process group destroyed.")

def main():
    """Main function to launch multi-GPU training."""
    logger.info("Starting multi-GPU training demo.")

    if not torch.cuda.is_available():
        logger.warning("No GPU found! Running on CPU.")
        num_gpus = 1 # Still use multiprocessing even on CPU for demo structure
    else:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs.")

    dataset = GINDataset(name="IMDBBINARY", self_loop=False)
    mp.spawn(run, args=(num_gpus, dataset), nprocs=num_gpus)
    logger.info("Multi-GPU training demo finished.")

if __name__ == "__main__":
    main()