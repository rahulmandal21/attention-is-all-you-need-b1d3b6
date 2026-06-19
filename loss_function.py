import torch
import torch.nn as nn

class SequenceToSequenceLossFunction(nn.Module):
    """
    A PyTorch module that calculates the cross-entropy loss for sequence-to-sequence models.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1, ignore_index: int = -100):
        """
        Initializes the SequenceToSequenceLossFunction module.

        Args:
        - num_classes (int): The number of classes in the classification problem.
        - smoothing (float, optional): The label smoothing factor. Defaults to 0.1.
        - ignore_index (int, optional): The index of the padding token to ignore. Defaults to -100.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=ignore_index)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross-entropy loss between the predictions and targets.

        Args:
        - predictions (torch.Tensor): The output logits of the model.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The calculated cross-entropy loss.
        """
        return self.criterion(predictions.view(-1, self.num_classes), targets.view(-1))

if __name__ == "__main__":
    # Create a dummy model and data
    model = nn.Linear(10, 10)
    input_data = torch.randn(1, 10)
    targets = torch.randint(0, 10, (1, 10))

    # Create an instance of the SequenceToSequenceLossFunction
    loss_function = SequenceToSequenceLossFunction(num_classes=10)

    # Calculate the loss
    predictions = model(input_data)
    loss = loss_function(predictions, targets)

    # Print the calculated loss
    print(loss.item())