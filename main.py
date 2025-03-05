import os
from loguru import logger
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import Adam

from mygo.datasets import load_dataset, DGLDatasetWrapper
from mygo.models import GINModel
from mygo.trainers import DistributedTrainer
from mygo.utils import TrainingArguments, init_process_group
from mygo.samplers import create_sampler



def main_process(rank, world_size, args):
    """Main process for each rank in distributed training."""
    init_process_group(world_size, rank) # Initialize process group at the beginning

    dataset = load_dataset(name="IMDBBINARY", self_loop=False)
    sampler = create_sampler()

    train_loader = sampler.get_train_dataloader(dataset, args.batch_size, world_size > 1, args.seed)
    val_loader = sampler.get_eval_dataloader(dataset, args.batch_size)
    test_loader = sampler.get_test_dataloader(dataset, args.batch_size)

    torch.manual_seed(args.seed)
    model = GINModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        args=args,
        rank=rank,
        world_size=world_size,
        sampler=sampler,
    )
    trainer.run()

def main():
    """Main entry point for training."""
    logger.info("Starting DGL Trainer demo.")

    if not torch.cuda.is_available():
        logger.warning("No GPU found! Running on CPU.")
        num_gpus = 1
    else:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs.")

    args = TrainingArguments()
    world_size = num_gpus
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size)
    logger.info("DGL Trainer demo finished.")

if __name__ == "__main__":
    main()