import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from enhanced_cs.LG_2507.configs import ProjectConfig

config = ProjectConfig()

tmp_dir = tempfile.mkdtemp()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


class Utils:
    @classmethod
    def validate_input(cls, data: Union[Tensor, np.ndarray], dim: int = 2) -> None:
        """
        Validate the input data for consistency and correctness.

        Args:
            data (Union[Tensor, np.ndarray]): Input data to be validated.
            dim (int, optional): Expected dimensionality of the data. Defaults to 2.

        Raises:
            ValueError: If the input data is not a tensor or numpy array, or has incorrect dimensionality.
        """
        if not isinstance(data, (Tensor, np.ndarray)):
            raise ValueError("Input data must be a tensor or numpy array.")

        if data.dim() != dim:
            raise ValueError(f"Expected {dim}D data, but got {data.dim()}D data.")

    @classmethod
    def to_local_device(cls, data: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Transfer the input data to the local device.

        Args:
            data (Union[Tensor, np.ndarray]): Input data to be transferred.

        Returns:
            Tensor: Data on the local device.
        """
        return data.to(config.device)

    @classmethod
    def setup_random_seed(cls, seed: Optional[int] = None) -> int:
        """
        Set up a random seed for reproducibility.

        Args:
            seed (int, optional): Specific seed to use. If None, a random seed is generated. Defaults to None.

        Returns:
            int: The random seed being used.
        """
        if seed is None:
            seed = np.random.randint(1, 10000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Random seed set to {seed}.")
        return seed

    @classmethod
    def save_checkpoint(cls, model, optimizer, epoch, save_path: str) -> None:
        """
        Save a checkpoint of the model and optimizer during training.

        Args:
            model: Torch model to be saved.
            optimizer: Optimizer to be saved.
            epoch (int): Current epoch number.
            save_path (str): Path to save the checkpoint.

        Raises:
            ValueError: If the model or optimizer cannot be saved.
        """
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        try:
            torch.save(state, save_path)
            logger.info(f"Checkpoint saved at epoch {epoch} to {save_path}")
        except Exception as e:
            raise ValueError("Unable to save checkpoint.") from e

    @classmethod
    def load_checkpoint(cls, load_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint and return the model and optimizer states.

        Args:
            load_path (str): Path to the checkpoint file.

        Returns:
            Dict[str, Any]: Dictionary containing model and optimizer states.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
            ValueError: If the checkpoint cannot be loaded.
        """
        try:
            checkpoint = torch.load(load_path)
            logger.info(f"Checkpoint loaded from {load_path}")
            return checkpoint
        except FileNotFoundError as e:
            raise FileNotFoundError("Checkpoint file not found.") from e
        except Exception as e:
            raise ValueError("Unable to load checkpoint.") from e

    @classmethod
    def save_metrics(cls, metrics: pd.DataFrame, save_path: str) -> None:
        """
        Save the metrics to a CSV file.

        Args:
            metrics (pd.DataFrame): DataFrame containing the metrics.
            save_path (str): Path to save the metrics file.

        Raises:
            ValueError: If the metrics cannot be saved.
        """
        try:
            metrics.to_csv(save_path, index=False)
            logger.info(f"Metrics saved to {save_path}")
        except Exception as e:
            raise ValueError("Unable to save metrics.") from e


class DataProcessor:
    def __init__(self, data: Union[Tensor, np.ndarray]):
        self.data = data
        self.device = config.device

    def to_local_device(self) -> None:
        """Transfer the data to the local device."""
        self.data = Utils.to_local_device(self.data)

    def preprocess(self) -> None:
        """Placeholder for data preprocessing steps."""
        raise NotImplementedError("Data preprocessing not implemented.")

    def standardize(self) -> None:
        """Standardize the data to have zero mean and unit variance."""
        self.data = (self.data - self.data.mean()) / self.data.std()

    def split(self, train_ratio: float = 0.8) -> Tuple[Tensor, Tensor]:
        """
        Split the data into training and validation sets.

        Args:
            train_ratio (float, optional): Ratio of data to use for training. Defaults to 0.8.

        Returns:
            Tuple[Tensor, Tensor]: Training and validation data.
        """
        num_samples = len(self.data)
        split_index = int(num_samples * train_ratio)
        train_data = self.data[:split_index]
        val_data = self.data[split_index:]
        return train_data, val_data


class ModelTrainer:
    def __init__(
        self,
        model,
        train_data: Tensor,
        val_data: Tensor,
        optimizer,
        loss_fn,
        metrics: List[Callable],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device

    def train(self, epochs: int = 100, checkpoint_path: str = None) -> pd.DataFrame:
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 100.
            checkpoint_path (str, optional): Path to save checkpoints. If None, no checkpoints are saved. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing training metrics.
        """
        self.model.train()
        metrics_df = pd.DataFrame()

        for epoch in range(epochs):
            # Training steps
            # ...

            # Validation steps
            # ...

            # Save checkpoint
            if checkpoint_path:
                Utils.save_checkpoint(self.model, self.optimizer, epoch, checkpoint_path)

        return metrics_df


class ModelEvaluator:
    def __init__(self, model, data: Tensor, loss_fn, metrics: List[Callable], device: str = "cpu"):
        self.model = model.to(device)
        self.data = data.to(device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the model on the given data.

        Returns:
            pd.DataFrame: DataFrame containing evaluation metrics.
        """
        self.model.eval()
        # Evaluation steps
        # ...

        return metrics_df


def velocity_threshold(data: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Apply the velocity threshold algorithm to the data.

    Args:
        data (Tensor): Input data.
        threshold (float, optional): Velocity threshold value. Defaults to 0.5.

    Returns:
        Tensor: Processed data after applying the velocity threshold.
    """
    # Implement the velocity-threshold algorithm
    # ...

    return processed_data


def flow_theory(data: Tensor, alpha: float = 0.1) -> Tensor:
    """
    Apply the flow theory algorithm to the data.

    Args:
        data (Tensor): Input data.
        alpha (float, optional): Flow theory parameter. Defaults to 0.1.

    Returns:
        Tensor: Processed data after applying flow theory.
    """
    # Implement the flow theory algorithm
    # ...

    return processed_data


def save_results(data: Dict[str, Any], save_path: str) -> None:
    """
    Save the results to a JSON file.

    Args:
        data (Dict[str, Any]): Dictionary containing the results.
        save_path (str): Path to save the results file.
    """
    import json

    with open(save_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Results saved to {save_path}")


def load_results(load_path: str) -> Dict[str, Any]:
    """
    Load the results from a JSON file.

    Args:
        load_path (str): Path to the results file.

    Returns:
        Dict[str, Any]: Dictionary containing the loaded results.
    """
    import json

    with open(load_path, "r") as f:
        data = json.load(f)
    logger.info(f"Results loaded from {load_path}")
    return data