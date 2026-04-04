# data_collector.py
import numpy as np
import pickle
import os
from config import Config
from typing import List, Tuple, Dict, Any
import random

class NVSieveDataCollector:
    """
    Collect training data for Nguyen-Vidick Sieve
    """
    def __init__(self, max_samples=Config.MAX_TRAINING_SAMPLES):
        self.training_data = []
        self.max_samples = max_samples
        self.samples_collected = 0
        
    def extract_features(self, v: np.ndarray, c: np.ndarray, 
                        R: float, gamma: float) -> np.ndarray:
        """
        Extract training features for (v, c) pair
        """
        v_norm = np.linalg.norm(v)
        c_norm = np.linalg.norm(c)
        
        # Basic features
        features = np.zeros(Config.INPUT_DIM)
        
        # 1. Norm features
        features[0] = v_norm
        features[1] = c_norm
        features[2] = abs(v_norm - c_norm)
        
        # 2. Geometric features
        dot_product = np.dot(v, c)
        features[3] = dot_product
        
        # 3. Angle features (if norms are non-zero)
        if v_norm > 0 and c_norm > 0:
            cos_sim = dot_product / (v_norm * c_norm)
            # Cosine of angle
            features[4] = cos_sim
            # Normalized angle (between 0 and 1)
            angle = np.arccos(np.clip(cos_sim, -1, 1))
            features[5] = angle / np.pi
        else:
            features[4] = 0
            features[5] = 0
            
        return features
    
    def record_match_attempt(self, v: np.ndarray, center_list: List[np.ndarray],
                           matched_center_idx: int, R: float, gamma: float) -> None:
        """
        Record result of a match attempt
        matched_center_idx: index of successfully matched center, -1 means no match
        """
        if self.samples_collected >= self.max_samples:
            return
            
        # Create samples for each checked center
        for i, c in enumerate(center_list):
            features = self.extract_features(v, c, R, gamma)
            
            # Label: 1 for successful match, 0 for failure
            label = 1 if i == matched_center_idx else 0
            
            self.training_data.append({
                'features': features,
                'label': label,
                'v_norm': np.linalg.norm(v),
                'c_norm': np.linalg.norm(c),
                'R': R,
                'gamma': gamma
            })
            
            self.samples_collected += 1
            if self.samples_collected >= self.max_samples:
                break
    
    def record_no_match(self, v: np.ndarray, center_list: List[np.ndarray],
                       R: float, gamma: float) -> None:
        """
        Record case where no match was found
        """
        self.record_match_attempt(v, center_list, -1, R, gamma)
    
    def get_training_dataset(self, positive_ratio=0.3):
        """Get balanced training dataset"""
        if not self.training_data:
            return None, None
        
        # Separate positive and negative samples
        positive_samples = [s for s in self.training_data if s['label'] == 1]
        negative_samples = [s for s in self.training_data if s['label'] == 0]
        
        print(f"Positive samples: {len(positive_samples)}, Negative samples: {len(negative_samples)}")
        
        # Oversample positive samples
        if len(positive_samples) < len(negative_samples) * positive_ratio:
            # Need to oversample positive samples
            target_positive = int(len(negative_samples) * positive_ratio)
            repeat_times = target_positive // len(positive_samples)
            remainder = target_positive % len(positive_samples)
            
            oversampled_positive = []
            for i in range(repeat_times):
                oversampled_positive.extend(positive_samples)
            oversampled_positive.extend(positive_samples[:remainder])
            
            positive_samples = oversampled_positive
        
        # Undersample negative samples
        target_negative = int(len(positive_samples) / positive_ratio)
        if len(negative_samples) > target_negative:
            negative_samples = random.sample(negative_samples, target_negative)
        
        # Combine dataset
        balanced_data = positive_samples + negative_samples
        random.shuffle(balanced_data)
        
        features = np.array([sample['features'] for sample in balanced_data])
        labels = np.array([sample['label'] for sample in balanced_data])
        
        print(f"After balancing: Positive={np.sum(labels)}, Negative={len(labels)-np.sum(labels)}")
        
        return features, labels
    
    def save_dataset(self, filename: str) -> None:
        """Save dataset"""
        # Simple fix: if path contains directory, create directory
        if '/' in filename:
            try:
                dirname = filename[:filename.rfind('/')]
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
            except:
                pass  # If creation fails, continue trying to save
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.training_data, f)
            print(f"Dataset saved to {filename}, total {len(self.training_data)} samples")
        except Exception as e:
            print(f"Failed to save dataset: {e}")
            # Try to save to current directory
            simple_name = filename.split('/')[-1] if '/' in filename else filename
            try:
                with open(simple_name, 'wb') as f:
                    pickle.dump(self.training_data, f)
                print(f"Dataset saved to {simple_name}")
            except Exception as e2:
                print(f"Failed to save to current directory too: {e2}")
    
    def load_dataset(self, filename: str) -> None:
        """Load dataset"""
        with open(filename, 'rb') as f:
            self.training_data = pickle.load(f)
        print(f"Loaded {len(self.training_data)} samples from {filename}")