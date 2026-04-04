# ai_enhanced_sieve.py
import numpy as np
from numpy.linalg import norm
from data_collector import NVSieveDataCollector
from model import CenterMatchMLP
import torch
from typing import List, Tuple, Optional
from config import Config
import random

class AIEnhancedNVSieve:
    """
    AI-enhanced Nguyen-Vidick Sieve
    """
    
    def __init__(self, model_path: str = None, 
                 top_k: int = Config.TOP_K,
                 use_ai: bool = True,
                 collect_data: bool = False):
        """
        Initialize
        
        Args:
            model_path: Pre-trained model path
            top_k: Number of centers to check per vector
            use_ai: Whether to use AI prediction
            collect_data: Whether to collect training data
        """
        self.top_k = top_k
        self.use_ai = use_ai
        self.collect_data = collect_data
        
        # Initialize data collector
        if collect_data:
            self.data_collector = NVSieveDataCollector()
        
        # Load AI model
        if use_ai and model_path:
            self.model = self._load_model(model_path)
        elif use_ai:
            # If no model path provided, use randomly initialized model
            self.model = CenterMatchMLP()
            self.model.eval()
        else:
            self.model = None
    
    def _load_model(self, model_path: str) -> CenterMatchMLP:
        """Load pre-trained model"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model based on saved configuration
        input_dim = checkpoint['model_config']['input_dim']
        hidden_dim = checkpoint['model_config']['hidden_dim']
        
        model = CenterMatchMLP(input_dim, hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded AI model from {model_path}")
        return model
    
    def _extract_features_batch(self, v: np.ndarray, 
                              center_list: List[np.ndarray],
                              R: float, gamma: float) -> np.ndarray:
        """
        Extract features for (v, c) pairs in batch
        """
        features_list = []
        v_norm = norm(v)
        
        for c in center_list:
            c_norm = norm(c)
            
            features = np.zeros(Config.INPUT_DIM)
            features[0] = v_norm
            features[1] = c_norm
            features[2] = abs(v_norm - c_norm)
            
            dot_product = np.dot(v, c)
            features[3] = dot_product
            
            if v_norm > 0 and c_norm > 0:
                cos_sim = dot_product / (v_norm * c_norm)
                features[4] = cos_sim
                angle = np.arccos(np.clip(cos_sim, -1, 1))
                features[5] = angle / np.pi
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def _predict_top_centers(self, v: np.ndarray, 
                           center_list: List[np.ndarray],
                           R: float, gamma: float) -> List[int]:
        """
        Predict top_k most likely center indices to match
        """
        if not center_list or self.model is None:
            return list(range(min(self.top_k, len(center_list))))
        
        # Extract features in batch
        features = self._extract_features_batch(v, center_list, R, gamma)
        
        # Predict match probabilities
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            probabilities = self.model(features_tensor).numpy()
        
        # Get top_k indices with highest probabilities
        top_indices = np.argsort(probabilities)[-self.top_k:][::-1]
        
        return top_indices.tolist()
    
    def _heuristic_top_centers(self, v: np.ndarray,
                             center_list: List[np.ndarray],
                             R: float, gamma: float) -> List[int]:
        """
        Heuristic method to select top_k centers (used when no model available)
        """
        if not center_list:
            return []
        
        # Simple heuristic: prefer centers with similar norms
        v_norm = norm(v)
        scores = []
        
        for i, c in enumerate(center_list):
            c_norm = norm(c)
            norm_diff = abs(v_norm - c_norm)
            # Smaller norm difference gets higher priority
            scores.append((i, -norm_diff))  # Negative sign because we want descending order
        
        # Sort by score, select top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:self.top_k]]
        
        return top_indices
    
    def enhanced_lattice_sieve(self, S: List[np.ndarray], 
                             gamma: float) -> Tuple[List[np.ndarray], int, float]:
        """
        Single step of AI-enhanced sieve
        
        Returns:
            S_p: Next round vector list
            checked_pairs: Number of center pairs checked
            avg_checked: Average number of centers checked per vector
        """
        R = sum(norm(v) for v in S) / len(S)
        gR = gamma * R
        
        C = []  # Center list
        S_p = []  # Next round vectors
        
        total_checked = 0
        found_matches = 0
        
        for v in S:
            if norm(v) <= gR:
                S_p.append(v)
            else:
                if not C:
                    C.append(v)
                    continue
                
                # Select centers to check
                if self.use_ai and self.model:
                    center_indices = self._predict_top_centers(v, C, R, gamma)
                else:
                    center_indices = self._heuristic_top_centers(v, C, R, gamma)
                
                # Record data collection
                if self.collect_data:
                    all_center_indices = list(range(len(C)))
                
                # Check selected centers
                matched = False
                matched_idx = -1
                
                for idx in center_indices:
                    if idx < len(C):
                        c = C[idx]
                        total_checked += 1
                        
                        if norm(v - c) <= gR:
                            S_p.append(v - c)
                            matched = True
                            matched_idx = idx
                            found_matches += 1
                            break
                
                # Record training data
                if self.collect_data:
                    if matched:
                        self.data_collector.record_match_attempt(v, C, matched_idx, R, gamma)
                    else:
                        self.data_collector.record_no_match(v, C, R, gamma)
                
                # If no match found, add v as new center
                if not matched:
                    C.append(v)
        
        # Calculate statistics
        avg_checked = total_checked / len(S) if S else 0
        
        return S_p, total_checked, avg_checked
    
    def run(self, S: List[np.ndarray], gamma: float, 
            max_iterations: int = 100) -> np.ndarray:
        """
        Run complete AI-enhanced sieve
        
        Args:
            S: Initial vector list
            gamma: Sieve parameter
            max_iterations: Maximum iterations
            
        Returns:
            Found shortest vector
        """
        S_0 = S.copy()
        iteration = 0
        
        print("Starting AI-enhanced sieve...")
        print(f"Initial vectors: {len(S)}, dimension: {S[0].shape[0]}")
        
        while len(S) > 0 and iteration < max_iterations:
            S_p, checked_pairs, avg_checked = self.enhanced_lattice_sieve(S, gamma)
            
            # Remove zero vectors
            S_p = [v for v in S_p if norm(v) > 1e-10]
            
            S = S_p
            iteration += 1
            
            if len(S) > 0:
                min_norm = norm(min(S, key=lambda v: norm(v)))
                print(f"\rIteration {iteration}: vectors={len(S)}, "
                      f"min norm={min_norm:.4f}, "
                      f"avg centers checked={avg_checked:.2f}", end="")
            
            if iteration >= max_iterations:
                print(f"\nReached maximum iterations {max_iterations}")
        
        # Find shortest vector from original set and final set
        if len(S) > 0:
            result = min(S, key=lambda v: norm(v))
        else:
            result = min(S_0, key=lambda v: norm(v))
        
        print(f"\nFound shortest vector, norm: {norm(result):.4f}")
        
        # Save collected data
        if self.collect_data and hasattr(self, 'data_collector'):
            self.data_collector.save_dataset(f"nv_sieve_data_{len(S_0[0])}d.pkl")
        
        return result