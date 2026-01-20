import numpy as np
import time
import random
from engine import ChessEngine
from rl_learner import PolicyValueNetwork, move_to_index
from supervised_learner import PositionEncoder
from mcts import MCTS

class SelfPlayManager:
    def __init__(self, num_simulations=50):
        self.engine = ChessEngine()
        self.model = PolicyValueNetwork()
        self.encoder = PositionEncoder()
        self.num_simulations = num_simulations
        self.memory = [] # Store (state, mcts_probs, result)

    def play_game(self):
        self.engine.reset()
        game_history = [] # (features, mcts_probs, turn)
        
        print("[Self-Play] Starting new game...")
        move_count = 0
        
        while self.engine.game_active and move_count < 100: # Limit to 100 moves
            mcts = MCTS(self.model, self.engine, self.encoder)
            # Perform MCTS search
            move_probs_dict = mcts.search(self.num_simulations)
            
            # Convert mcts_probs_dict to a full 4096 distribution
            pi = np.zeros(64 * 64)
            legal_moves = []
            
            for m_str, prob in move_probs_dict.items():
                # Reconstruct move from string or just get from engine
                # For this implementation, we search the legal moves
                # m_str is "fr,fc-tr,tc"
                pass
            
            # Simpler approach: Select move based on MCTS visit counts
            best_m_str = max(move_probs_dict.items(), key=lambda x: x[1])[0]
            
            # Find the actual move object in engine
            all_legal = self.engine.get_all_legal_moves(self.engine.turn)
            selected_move = None
            for m in all_legal:
                m_str_check = f"{m['from']}-{m['to']}"
                if m_str_check == best_m_str:
                    selected_move = m
                    break
            
            if not selected_move:
                break
                
            # Store data for training
            features = self.encoder.encode_position(self.engine.board, self.engine.turn)
            # Create a target policy from MCTS probabilities
            target_pi = np.zeros(64 * 64)
            for m_str, prob in move_probs_dict.items():
                # We need move_to_index(m)
                # This is inefficient but works for a demo
                for m in all_legal:
                    if f"{m['from']}-{m['to']}" == m_str:
                        target_pi[move_to_index(m)] = prob
                        break
            
            game_history.append((features, target_pi, self.engine.turn))
            
            # Make the move
            self.engine.make_move(selected_move)
            move_count += 1
            # print(f"  Move {move_count}: {best_m_str}")

        # Determine result
        result = 0
        if self.engine.is_checkmate('w'):
            result = -1 # Black wins
        elif self.engine.is_checkmate('b'):
            result = 1 # White wins
        
        print(f"[Self-Play] Game Over in {move_count} moves. Result: {result}")
        
        # Add to memory with results from perspective of each turn
        for features, target_pi, turn in game_history:
            # result is from white's perspective
            v_target = result if turn == 'w' else -result
            self.memory.append((features, target_pi, v_target))

    def train(self, batch_size=32, epochs=1):
        if len(self.memory) < batch_size:
            return
            
        print(f"[Training] Training on {len(self.memory)} states...")
        
        # Shuffle memory
        random.shuffle(self.memory)
        
        for _ in range(epochs):
            for i in range(0, len(self.memory), batch_size):
                batch = self.memory[i:i+batch_size]
                X = np.array([item[0] for item in batch])
                target_p = np.array([item[1] for item in batch])
                target_v = np.array([[item[2]] for item in batch])
                
                self.model.train_step(X, target_p, target_v)
        
        self.model.save_model()
        # Clear memory after training or use a sliding window
        self.memory = self.memory[-1000:] # Keep last 1000

if __name__ == "__main__":
    manager = SelfPlayManager(num_simulations=20)
    for i in range(10): # Play 10 games
        print(f"\n--- Iteration {i+1} ---")
        manager.play_game()
        manager.train()
