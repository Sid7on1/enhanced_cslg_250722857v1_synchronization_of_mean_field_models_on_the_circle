import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class MeanFieldModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_heads, num_layers):
        super(MeanFieldModel, self).__init__()
        self.self_attention = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.self_attention(x, x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

class KuramotoModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_heads, num_layers):
        super(KuramotoModel, self).__init__()
        self.self_attention = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.self_attention(x, x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        outputs = self.encoder(x)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.classifier(x)
        return x

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

    def _create_model(self):
        if self.config.model_type == 'mean_field':
            return MeanFieldModel(self.config.num_classes, self.config.hidden_dim, self.config.num_heads, self.config.num_layers)
        elif self.config.model_type == 'kuramoto':
            return KuramotoModel(self.config.num_classes, self.config.hidden_dim, self.config.num_heads, self.config.num_layers)
        elif self.config.model_type == 'transformer':
            return TransformerModel(self.config.num_classes, self.config.hidden_dim, self.config.num_heads, self.config.num_layers)
        else:
            raise ValueError('Invalid model type')

    def _create_optimizer(self):
        if self.config.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError('Invalid optimizer')

    def train(self, train_loader, val_loader):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            for batch in train_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                for batch in val_loader:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / len(val_loader.dataset)
                logging.info(f'Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader)}, Val Acc: {accuracy:.4f}')
            self.model.train()

    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for batch in test_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            accuracy = correct / len(test_loader.dataset)
            logging.info(f'Test Acc: {accuracy:.4f}')

def main():
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    config = pd.read_json(args.config)
    train_pipeline = TrainingPipeline(config)
    train_loader, val_loader, test_loader = train_pipeline._create_data_loaders()
    train_pipeline.train(train_loader, val_loader)
    train_pipeline.evaluate(test_loader)

if __name__ == '__main__':
    main()