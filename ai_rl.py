import time
import numpy as np
from engine import ChessEngine
from rl_learner import PolicyValueNetwork, move_to_index
from supervised_learner import PositionEncoder
from mcts import MCTS

class RLChessAI:
    def __init__(self, engine):
        self.engine = engine
        self.model = PolicyValueNetwork()
        self.encoder = PositionEncoder()
        self.num_simulations = 40 # Adjust for speed/strength
        
    def get_best_move(self):
        """Get best move using MCTS and Policy-Value Network"""
        if not self.engine.game_active:
            return None
            
        print(f"[RL-AI] Searching with {self.num_simulations} simulations...")
        mcts = MCTS(self.model, self.engine, self.encoder)
        move_probs_dict = mcts.search(self.num_simulations)
        
        if not move_probs_dict:
            return None
            
        # Select move with highest visit count
        best_m_str = max(move_probs_dict.items(), key=lambda x: x[1])[0]
        
        # Convert string back to move object
        all_legal = self.engine.get_all_legal_moves(self.engine.turn)
        for m in all_legal:
            m_str = f"{m['from']}-{m['to']}"
            if m_str == best_m_str:
                return m
                
        return None
