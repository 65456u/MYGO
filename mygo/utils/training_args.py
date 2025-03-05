import os
import torch

class TrainingArguments:
    def __init__(
        self,
        output_dir="./output",
        num_train_epochs=500,
        learning_rate=0.01,
        batch_size=512,
        seed=0,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ddp_backend="nccl",
        log_dir='./runs',
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.device = device
        self.ddp_backend = ddp_backend
        self.log_dir = log_dir

        os.makedirs(self.output_dir, exist_ok=True)