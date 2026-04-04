# trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from model import CenterMatchMLP, EarlyStopping
from config import Config

class ModelTrainer:
    """Model Trainer"""
    
    def __init__(self, model=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or CenterMatchMLP().to(self.device)
        
    def prepare_dataset(self, features, labels, test_size=0.2):
        """Prepare training and validation datasets"""
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, epochs=Config.EPOCHS, 
              batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE):
        """Train model"""
        _, labels = train_dataset.tensors
        pos_weight = (len(labels) - labels.sum()) / labels.sum()
        pos_weight = torch.tensor([pos_weight]).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Early stopping mechanism
        early_stopping = EarlyStopping(patience=10)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            # Calculate average loss and accuracy
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.network[0].in_features,
                'hidden_dim': Config.HIDDEN_DIM
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {path}")