import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

class DataPreprocessing:
    """
    A class used to preprocess data for the transformer model.

    Attributes:
    ----------
    d_model : int
        The dimension of the model.
    max_len : int
        The maximum length of the input sequences.
    padding_value : int
        The value used for padding.

    Methods:
    -------
    __init__(d_model: int, max_len: int, padding_value: int)
        Initializes the DataPreprocessing class.
    create_positional_encoding(max_len: int) -> torch.Tensor
        Creates the positional encoding.
    create_embedding(num_embeddings: int, embedding_dim: int) -> nn.Embedding
        Creates the learned embeddings.
    pad_sequences(sequences: list) -> torch.Tensor
        Pads the input sequences to the same length.
    preprocess_data(sequences: list) -> torch.Tensor
        Preprocesses the data by creating the positional encoding, learned embeddings, and padding the sequences.
    """

    def __init__(self, d_model: int, max_len: int, padding_value: int):
        """
        Initializes the DataPreprocessing class.

        Args:
        ----
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the input sequences.
        padding_value (int): The value used for padding.
        """
        self.d_model = d_model
        self.max_len = max_len
        self.padding_value = padding_value

    def create_positional_encoding(self, max_len: int) -> torch.Tensor:
        """
        Creates the positional encoding.

        Args:
        ----
        max_len (int): The maximum length of the input sequences.

        Returns:
        -------
        torch.Tensor: The positional encoding.
        """
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def create_embedding(self, num_embeddings: int, embedding_dim: int) -> nn.Embedding:
        """
        Creates the learned embeddings.

        Args:
        ----
        num_embeddings (int): The number of embeddings.
        embedding_dim (int): The dimension of the embeddings.

        Returns:
        -------
        nn.Embedding: The learned embeddings.
        """
        return nn.Embedding(num_embeddings, embedding_dim)

    def pad_sequences(self, sequences: list) -> torch.Tensor:
        """
        Pads the input sequences to the same length.

        Args:
        ----
        sequences (list): The input sequences.

        Returns:
        -------
        torch.Tensor: The padded sequences.
        """
        return pad_sequence(sequences, batch_first=True, padding_value=self.padding_value)

    def preprocess_data(self, sequences: list) -> torch.Tensor:
        """
        Preprocesses the data by creating the positional encoding, learned embeddings, and padding the sequences.

        Args:
        ----
        sequences (list): The input sequences.

        Returns:
        -------
        torch.Tensor: The preprocessed data.
        """
        pe = self.create_positional_encoding(self.max_len)
        embedding = self.create_embedding(num_embeddings=max(sequences) + 1, embedding_dim=self.d_model)
        padded_sequences = self.pad_sequences(sequences)
        embedded_sequences = embedding(padded_sequences) * math.sqrt(self.d_model)
        return embedded_sequences + pe[:embedded_sequences.size(0), :]

class TokenizedTextDataset(Dataset):
    """
    A class used to create a dataset for the tokenized text sequences.

    Attributes:
    ----------
    sequences : list
        The tokenized text sequences.

    Methods:
    -------
    __init__(sequences: list)
        Initializes the TokenizedTextDataset class.
    __len__() -> int
        Returns the length of the dataset.
    __getitem__(idx: int) -> torch.Tensor
        Returns the sequence at the given index.
    """

    def __init__(self, sequences: list):
        """
        Initializes the TokenizedTextDataset class.

        Args:
        ----
        sequences (list): The tokenized text sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        -------
        int: The length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the sequence at the given index.

        Args:
        ----
        idx (int): The index of the sequence.

        Returns:
        -------
        torch.Tensor: The sequence at the given index.
        """
        return self.sequences[idx]

def collate_fn(batch):
    """
    A function used to collate the batches.

    Args:
    ----
    batch (list): The batch of sequences.

    Returns:
    -------
    torch.Tensor: The padded batch of sequences.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(d_model=512, max_len=100, padding_value=0)
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    preprocessed_data = data_preprocessing.preprocess_data(sequences)
    print(preprocessed_data)

    dataset = TokenizedTextDataset(sequences)
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)