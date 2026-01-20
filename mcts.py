import numpy as np
import math
import copy
from rl_learner import move_to_index

class MCTSNode:
    def __init__(self, state, parent=None, prior=0):
        self.state = state  # Board state (engine instance or FEN)
        self.parent = parent
        self.children = {}  # move_str -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.state_features = None # Cached features

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            # Upper Confidence Bound for Trees (PUCT variant)
            score = child.value + c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child

class MCTS:
    def __init__(self, model, engine, encoder, c_puct=1.4):
        self.model = model
        self.engine = engine
        self.encoder = encoder
        self.c_puct = c_puct

    def search(self, num_simulations):
        # Current state is root
        # Note: In real implementation, we should not copy the entire engine often.
        # But for this demo, we use engine.make_move/undo_move.
        
        root = MCTSNode(self._get_current_state())
        
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # 1. Select
            while node.children:
                move_str, node = node.select_child(self.c_puct)
                from_sq, to_sq = node.move
                move = {'from': from_sq, 'to': to_sq}
                self.engine.make_move(move)
                search_path.append(node)
            
            # 2. Expand & Evaluate
            features = self.encoder.encode_position(self.engine.board, self.engine.turn)
            policy, value = self.model.forward(features)
            value = value[0][0]
            
            legal_moves = self.engine.get_all_legal_moves(self.engine.turn)
            if not legal_moves or self.engine.is_draw():
                if self.engine.is_checkmate(self.engine.turn):
                    value = -1.0
                else:
                    value = 0.0
            else:
                for m in legal_moves:
                    m_str = f"{m['from']}-{m['to']}"
                    m_idx = move_to_index(m)
                    prior = policy[0][m_idx]
                    node.children[m_str] = MCTSNode(None, parent=node, prior=prior)
                    node.children[m_str].move = (m['from'], m['to'])
            
            # 3. Backpropagate
            curr_val = value
            for b_node in reversed(search_path):
                b_node.visit_count += 1
                b_node.value_sum += curr_val
                curr_val = -curr_val
                
            # Undo moves to restore engine state
            for _ in range(len(search_path) - 1):
                self.engine.undo_move()
                
        # Return action probabilities
        return self._get_action_probs(root)

    def _get_current_state(self):
        # Simplified: just a marker
        return "state"

    def _get_action_probs(self, root):
        probs = np.zeros(64 * 64)
        for m_str, child in root.children.items():
            # Real AlphaZero uses visit counts for probabilities
            # pi(a|s) = N(s,a)^(1/temp) / sum(N(s,b)^(1/temp))
            # temp=1 for now
            # Extract move from m_str "fr,fc-tr,tc"
            parts = m_str.split('-')
            # This is slow, better to store move in child
            # For brevity, let's assume we can reconstruct it or store it.
            # I'll modify MCTSNode to store move index.
            pass
        
        # Refined:
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
             # Fallback to priors
             return np.array([child.prior for child in root.children.values()])

        move_probs = {}
        for m_str, child in root.children.items():
            move_probs[m_str] = child.visit_count / total_visits
            
        return move_probs
