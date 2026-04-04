# enum_environment.py
import numpy as np
import os
import torch
import sys
sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False

class EnumEnvironment:
    def __init__(self, lattice, config):
        self.lattice = lattice
        self.config = config
        self.wrapper = lattice_env.RL_ENUM_Wrapper(lattice)
        
        # Fixed state dimension: 30
        self.state_space_dim = 30
        # Fixed action dimension: 11 (-5 to +5)
        self.action_space_dim = 11
        self.step_count = 0
    
    def reset(self, radius=None):
        """Reset the environment"""
        if radius is None:
            radius = self.config.radius
        
        self.wrapper.reset(radius)
        self.step_count = 0
        state_obj = self.wrapper.get_state()
        
        # Extract fixed-dimension features
        return self.extract_enum_features(state_obj)  # 注意：使用self.而不是state_obj.
    
    def step(self, action):
        """Execute one step"""
        self.step_count += 1
        
        # Action mapping: 0-10 -> -5 to +5
        # 注意：C++的decode_action期望的是偏移量，所以需要映射
        offset_action = action - 5  # 将0-10映射到-5到+5
        
        # Execute one ENUM step
        reward, done, info_str = self.wrapper.step(offset_action)
        
        # Get new state
        state_obj = self.wrapper.get_state()
        next_state = self.extract_enum_features(state_obj)  # 注意：使用self.
        
        # 解析信息并添加额外信息
        info = {
            'best_norm': state_obj.best_norm,
            'current_k': state_obj.current_k,
            'current_rho': state_obj.current_rho,
            'solved': state_obj.found_solution,
            'total_steps': self.step_count,
            'info_str': info_str
        }
        
        return next_state, reward, done, info

    def extract_enum_features(self, state_obj):
        """
        Extract features with robust numerical stability
        """
        features = []
        
        # 1. 添加更多安全检查
        if not hasattr(state_obj, 'current_k'):
            # 返回默认特征向量
            return np.zeros(self.state_space_dim, dtype=np.float32)
        
        # 2. 使用更安全的归一化
        # Current layer position
        if state_obj.num_rows > 0:
            normalized_k = float(state_obj.current_k) / float(state_obj.num_rows)
        else:
            normalized_k = 0.0
        features.append(np.clip(normalized_k, 0.0, 1.0))
        
        # Current ρ relative value
        if state_obj.radius > 0 and state_obj.radius > 1e-8:
            rho_ratio = float(state_obj.current_rho) / float(state_obj.radius)
            features.append(np.clip(rho_ratio, 0.0, 100.0))  # 限制范围
        else:
            features.append(0.0)
        
        # Search progress indicator
        if (state_obj.best_norm < 1e10 and state_obj.radius > 1e-8 and 
            state_obj.best_norm >= 0):
            progress = 1.0 - (float(state_obj.best_norm) / float(state_obj.radius))
            features.append(np.clip(progress, 0.0, 1.0))
        else:
            features.append(0.0)
        
        # Status flags
        features.append(1.0 if getattr(state_obj, 'has_solution', False) else 0.0)
        features.append(1.0 if getattr(state_obj, 'terminated', False) else 0.0)
        
        # 3. 当前层信息 - 更安全地处理
        k = int(state_obj.current_k)
        
        # Current center
        center_val = float(getattr(state_obj, 'current_center', 0.0))
        features.append(np.clip(center_val / 10.0, -10.0, 10.0))
        
        # GS norm
        if hasattr(state_obj, 'gs_norms') and state_obj.gs_norms:
            try:
                gs_list = [float(x) for x in state_obj.gs_norms]
                if len(gs_list) > 0 and k < len(gs_list):
                    base_val = max(abs(gs_list[0]), 1e-8)
                    gs_val = gs_list[k] / base_val
                    features.append(np.clip(gs_val, 0.0, 100.0))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Current coefficient
        if hasattr(state_obj, 'current_coeffs') and state_obj.current_coeffs:
            try:
                coeff_list = [float(x) for x in state_obj.current_coeffs]
                if len(coeff_list) > 0 and k < len(coeff_list):
                    coeff_val = coeff_list[k] / 10.0
                    features.append(np.clip(coeff_val, -10.0, 10.0))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # μ values
        if hasattr(state_obj, 'mu_values') and state_obj.mu_values:
            try:
                mu_list = [float(x) for x in state_obj.mu_values]
                if mu_list:
                    mu_avg = np.mean([abs(x) for x in mu_list if not np.isnan(x)])
                    features.append(float(np.clip(mu_avg, 0.0, 10.0)))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 4. 填充剩余特征
        # 确保特征维度固定为30
        while len(features) < self.state_space_dim:
            features.append(0.0)
        
        # 转换为数组
        features_array = np.array(features[:self.state_space_dim], dtype=np.float32)
        
        # 5. 最终检查和修复
        # 检查并修复NaN/Inf
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            print(f"??  Detected NaN/Inf in features, fixing...")
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 归一化特征
        if np.std(features_array) > 0:
            features_array = (features_array - np.mean(features_array)) / (np.std(features_array) + 1e-8)
        
        # 限制范围
        features_array = np.clip(features_array, -5.0, 5.0)
        
        return features_array

    def get_best_norm(self):
        """Get the best norm found so far"""
        state_obj = self.wrapper.get_state()
        return state_obj.best_norm
    
    def get_best_vector(self):
        """Get the best vector found so far"""
        return self.wrapper.get_best_vector()
    
    def is_terminated(self):
        """Check if the search has terminated"""
        state_obj = self.wrapper.get_state()
        return state_obj.terminated or self.wrapper.is_terminated()