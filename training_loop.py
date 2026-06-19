import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

class Trainer:
    """
    A class used to train a PyTorch model with a warmup schedule for the learning rate.

    Attributes:
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    dataloader : DataLoader
        The data loader for the training data.
    optimizer : nn.optim
        The optimizer for the model parameters.
    loss_fn : nn.Module
        The loss function for the model.
    device : torch.device
        The device to train the model on.
    d_model : int
        The dimension of the model.
    warmup_steps : int
        The number of warmup steps.
    max_grad_norm : float
        The maximum gradient norm.

    Methods:
    -------
    train_one_epoch()
        Trains the model for one epoch.
    get_lr()
        Gets the learning rate based on the warmup schedule.
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer: nn.optim, loss_fn: nn.Module, device: torch.device, d_model: int, warmup_steps: int, max_grad_norm: float = 1.0):
        """
        Initializes the Trainer class.

        Args:
        ----
        model (nn.Module): The PyTorch model to be trained.
        dataloader (DataLoader): The data loader for the training data.
        optimizer (nn.optim): The optimizer for the model parameters.
        loss_fn (nn.Module): The loss function for the model.
        device (torch.device): The device to train the model on.
        d_model (int): The dimension of the model.
        warmup_steps (int): The number of warmup steps.
        max_grad_norm (float, optional): The maximum gradient norm. Defaults to 1.0.
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.step_num = 0

    def train_one_epoch(self) -> float:
        """
        Trains the model for one epoch.

        Returns:
        -------
        float
            The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            self.step_num += 1
            self.optimizer.param_groups[0]['lr'] = self.get_lr()
        return total_loss / len(self.dataloader)

    def get_lr(self) -> float:
        """
        Gets the learning rate based on the warmup schedule.

        Returns:
        -------
        float
            The learning rate.
        """
        return self.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)


if __name__ == "__main__":
    # Create a dummy model, data loader, and optimizer
    model = nn.Linear(5, 3)
    dataloader = DataLoader([(torch.randn(5), torch.randn(3)) for _ in range(10)], batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 512
    warmup_steps = 1000
    max_grad_norm = 1.0

    # Create a Trainer instance and train the model for one epoch
    trainer = Trainer(model, dataloader, optimizer, loss_fn, device, d_model, warmup_steps, max_grad_norm)
    average_loss = trainer.train_one_epoch()
    print(f'Average loss: {average_loss}')