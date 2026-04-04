# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class CenterMatchMLP(nn.Module):
    """
    Simple MLP model to predict match probability for (v, c) pairs
    """
    def __init__(self, input_dim=Config.INPUT_DIM, 
                 hidden_dim=Config.HIDDEN_DIM, 
                 dropout=0.2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict_proba(self, features):
        """
        Predict probabilities
        features: numpy array or torch tensor
        """
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            predictions = self(features).cpu().numpy()
        return predictions


class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0