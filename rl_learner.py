import numpy as np
import os
import pickle
from supervised_learner import PositionEncoder

class PolicyValueNetwork:
    """
    AlphaZero-style Policy-Value Network implemented in NumPy
    - Policy head: Probability distribution over 4096 move squares (from*64 + to)
    - Value head: Scaled evaluation [-1, 1]
    """
    def __init__(self, input_size=256, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.policy_output_size = 64 * 64  # Simplified: from_sq * 64 + to_sq
        
        # Weights initialization (Xavier/Glorot)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Policy Head
        self.W_policy = np.random.randn(hidden_size, self.policy_output_size) * np.sqrt(2.0 / hidden_size)
        self.b_policy = np.zeros((1, self.policy_output_size))
        
        # Value Head
        self.W_value = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b_value = np.zeros((1, 1))
        
        self.model_file = "chess_rl_model.pkl"
        self.load_model()

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        """Forward pass returns (policy, value)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Policy head
        self.z_p = np.dot(self.a1, self.W_policy) + self.b_policy
        self.policy = self.softmax(self.z_p)
        
        # Value head
        self.z_v = np.dot(self.a1, self.W_value) + self.b_value
        self.value = self.tanh(self.z_v)
        
        return self.policy, self.value

    def train_step(self, x, target_policy, target_value, learning_rate=0.01):
        """
        One step of gradient descent
        target_policy: One-hot or distribution of moves (1, 4096)
        target_value: Game result [-1, 1] (1, 1)
        """
        m = x.shape[0]
        
        # 1. Forward
        p, v = self.forward(x)
        
        # 2. Gradients for Value Head (MSE loss)
        # Loss = (v - target_value)^2
        # dL/dv = 2(v - target_value)
        # dL/dz_v = dL/dv * dv/dz_v = 2(v - target_value) * (1 - tanh^2(z_v))
        dv = 2 * (v - target_value)
        dz_v = dv * (1 - np.square(v))
        dW_value = np.dot(self.a1.T, dz_v) / m
        db_value = np.sum(dz_v, axis=0, keepdims=True) / m
        
        # 3. Gradients for Policy Head (Cross-Entropy loss)
        # dL/dz_p = p - target_policy
        dz_p = (p - target_policy)
        dW_policy = np.dot(self.a1.T, dz_p) / m
        db_policy = np.sum(dz_p, axis=0, keepdims=True) / m
        
        # 4. Gradients for Shared Layer
        da1 = np.dot(dz_p, self.W_policy.T) + np.dot(dz_v, self.W_value.T)
        dz1 = da1 * (self.z1 > 0) # ReLU derivative
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 5. Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W_policy -= learning_rate * dW_policy
        self.b_policy -= learning_rate * db_policy
        self.W_value -= learning_rate * dW_value
        self.b_value -= learning_rate * db_value

    def save_model(self):
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'W1': self.W1, 'b1': self.b1,
                    'W_policy': self.W_policy, 'b_policy': self.b_policy,
                    'W_value': self.W_value, 'b_value': self.b_value
                }, f)
        except Exception as e:
            print(f"[RL] Error saving model: {e}")

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.W1 = data['W1']
                    self.b1 = data['b1']
                    self.W_policy = data['W_policy']
                    self.b_policy = data['b_policy']
                    self.W_value = data['W_value']
                    self.b_value = data['b_value']
                print(f"[RL] Model loaded from {self.model_file}")
            except Exception as e:
                print(f"[RL] Error loading model: {e}")

def move_to_index(move):
    """Convert engine move to 0-4095 index"""
    fr, fc = move['from']
    tr, tc = move['to']
    from_idx = fr * 8 + fc
    to_idx = tr * 8 + tc
    return from_idx * 64 + to_idx

def index_to_move(idx):
    """Convert 0-4095 index back to (from_sq, to_sq)"""
    from_idx = idx // 64
    to_idx = idx % 64
    return (from_idx // 8, from_idx % 8), (to_idx // 8, to_idx % 8)
