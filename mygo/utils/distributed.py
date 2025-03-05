import torch.distributed as dist # Import torch.distributed
from loguru import logger

def init_process_group(world_size, rank, host = "127.0.0.1", port = 12345, protocal = "tcp", backend = "nccl"):
    """Initialize the distributed process group."""
    dist.init_process_group(
        backend=backend,  # Use 'nccl' for multiple GPUs
        init_method=f"{protocal}://{host}:{port}", #"tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )
    logger.info(f"Rank {rank}: Process group initialized.")