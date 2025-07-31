import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def calculate_accuracy(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate accuracy score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: Accuracy score.
        """
        try:
            accuracy = accuracy_score(labels, predictions)
            logger.info(f"Accuracy: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return None

    def calculate_f1_score(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate F1 score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: F1 score.
        """
        try:
            f1 = f1_score(labels, predictions)
            logger.info(f"F1 score: {f1:.4f}")
            return f1
        except Exception as e:
            logger.error(f"Error calculating F1 score: {e}")
            return None

    def calculate_precision(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate precision score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: Precision score.
        """
        try:
            precision = precision_score(labels, predictions)
            logger.info(f"Precision: {precision:.4f}")
            return precision
        except Exception as e:
            logger.error(f"Error calculating precision: {e}")
            return None

    def calculate_recall(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate recall score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: Recall score.
        """
        try:
            recall = recall_score(labels, predictions)
            logger.info(f"Recall: {recall:.4f}")
            return recall
        except Exception as e:
            logger.error(f"Error calculating recall: {e}")
            return None

    def calculate_mean_field(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate mean field score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: Mean field score.
        """
        try:
            mean_field = np.mean(np.abs(np.array(predictions) - np.array(labels)))
            logger.info(f"Mean field: {mean_field:.4f}")
            return mean_field
        except Exception as e:
            logger.error(f"Error calculating mean field: {e}")
            return None

    def calculate_velocity_threshold(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate velocity threshold score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: Velocity threshold score.
        """
        try:
            velocity_threshold = np.mean(np.abs(np.diff(np.array(predictions))))
            logger.info(f"Velocity threshold: {velocity_threshold:.4f}")
            return velocity_threshold
        except Exception as e:
            logger.error(f"Error calculating velocity threshold: {e}")
            return None

    def calculate_flow_theory(self, predictions: List[int], labels: List[int]) -> float:
        """
        Calculate flow theory score.

        Args:
        predictions (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.

        Returns:
        float: Flow theory score.
        """
        try:
            flow_theory = np.mean(np.abs(np.array(predictions) - np.array(labels)) / np.mean(np.abs(np.diff(np.array(predictions)))))
            logger.info(f"Flow theory: {flow_theory:.4f}")
            return flow_theory
        except Exception as e:
            logger.error(f"Error calculating flow theory: {e}")
            return None

class Evaluation:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = EvaluationMetrics(model, tokenizer)

    def evaluate(self, inputs: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
        inputs (List[str]): List of input strings.
        labels (List[int]): List of true labels.

        Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
        """
        try:
            # Preprocess inputs
            inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True)

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model predictions
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            # Calculate evaluation metrics
            metrics = {
                "accuracy": self.metrics.calculate_accuracy(predictions, labels),
                "f1_score": self.metrics.calculate_f1_score(predictions, labels),
                "precision": self.metrics.calculate_precision(predictions, labels),
                "recall": self.metrics.calculate_recall(predictions, labels),
                "mean_field": self.metrics.calculate_mean_field(predictions, labels),
                "velocity_threshold": self.metrics.calculate_velocity_threshold(predictions, labels),
                "flow_theory": self.metrics.calculate_flow_theory(predictions, labels)
            }

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

def main():
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create evaluation object
    evaluation = Evaluation(model, tokenizer, device)

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into training and testing sets
    inputs, labels = data["text"], data["label"]
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    # Preprocess inputs
    inputs_train = tokenizer.batch_encode_plus(inputs_train, return_tensors="pt", padding=True, truncation=True)
    inputs_test = tokenizer.batch_encode_plus(inputs_test, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to device
    inputs_train = {k: v.to(device) for k, v in inputs_train.items()}
    inputs_test = {k: v.to(device) for k, v in inputs_test.items()}

    # Get model predictions
    outputs_train = model(**inputs_train)
    predictions_train = torch.argmax(outputs_train.logits, dim=1).cpu().numpy()
    outputs_test = model(**inputs_test)
    predictions_test = torch.argmax(outputs_test.logits, dim=1).cpu().numpy()

    # Calculate evaluation metrics
    metrics_train = evaluation.metrics.evaluate(inputs_train["input_ids"].tolist(), labels_train.tolist())
    metrics_test = evaluation.metrics.evaluate(inputs_test["input_ids"].tolist(), labels_test.tolist())

    # Print evaluation metrics
    logger.info(f"Training metrics: {metrics_train}")
    logger.info(f"Testing metrics: {metrics_test}")

if __name__ == "__main__":
    main()