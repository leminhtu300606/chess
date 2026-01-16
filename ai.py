import random
import time
from config import *

class ChessAI:
    def __init__(self, engine):
        self.engine = engine
        self.difficulty = 'medium'
        self.max_time = 1.5  # Max seconds per move

    def get_best_move(self):
        if not self.engine.game_active:
            return None
        
        # Opening book
        book_move = self.get_opening_move()
        if book_move:
            return book_move
        
        # Set depth based on difficulty
        depth = {'easy': 1, 'medium': 2, 'hard': 2}.get(self.difficulty, 2)
        
        moves = self.engine.get_all_legal_moves(self.engine.turn)
        if not moves:
            return None
        
        # Sort moves: captures first for better pruning
        moves.sort(key=lambda m: self._move_score(m), reverse=True)
        
        best_move = moves[0]
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        self.start_time = time.time()
        
        for move in moves:
            # Time check
            if time.time() - self.start_time > self.max_time:
                break
            
            self.engine.make_move(move, switch_turn=True, record_history=True)
            val = self.minimax(depth - 1, alpha, beta, False)
            self.engine.undo_move()
            
            if val > best_value:
                best_value = val
                best_move = move
            alpha = max(alpha, best_value)
        
        return best_move

    def _move_score(self, move):
        # Prioritize captures and checks for move ordering
        score = 0
        target = self.engine.board[move['to'][0]][move['to'][1]]
        if target:
            score += PIECE_VALUES.get(target['type'], 0)
        return score

    def minimax(self, depth, alpha, beta, is_max):
        # Time limit check
        if time.time() - self.start_time > self.max_time:
            return self.evaluate()
        
        if depth == 0:
            return self.quiescence(alpha, beta, is_max, 2)

        current_turn = self.engine.turn
        moves = self.engine.get_all_legal_moves(current_turn)
        
        if not moves:
            if self.engine.is_in_check(current_turn):
                return -20000 if is_max else 20000
            return 0
        
        # Move ordering
        moves.sort(key=lambda m: self._move_score(m), reverse=True)
        
        if is_max:
            max_eval = -float('inf')
            for move in moves:
                self.engine.make_move(move, switch_turn=True, record_history=True)
                eval_score = self.minimax(depth - 1, alpha, beta, False)
                self.engine.undo_move()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                self.engine.make_move(move, switch_turn=True, record_history=True)
                eval_score = self.minimax(depth - 1, alpha, beta, True)
                self.engine.undo_move()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def quiescence(self, alpha, beta, is_max, depth):
        stand_pat = self.evaluate()
        
        if depth == 0:
            return stand_pat
        
        if is_max:
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if beta > stand_pat:
                beta = stand_pat

        current_turn = self.engine.turn
        moves = self.engine.get_all_legal_moves(current_turn)
        
        # Only captures
        captures = [m for m in moves if self.engine.board[m['to'][0]][m['to'][1]]]
        captures.sort(key=lambda m: self._move_score(m), reverse=True)
        
        for m in captures[:5]:  # Limit captures to check
            self.engine.make_move(m, switch_turn=True, record_history=True)
            score = self.quiescence(alpha, beta, not is_max, depth - 1)
            self.engine.undo_move()

            if is_max:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score

        return alpha if is_max else beta

    def evaluate(self):
        # Simple material evaluation
        score = 0
        for r in range(8):
            for c in range(8):
                piece = self.engine.board[r][c]
                if piece:
                    val = PIECE_VALUES.get(piece['type'], 0)
                    # Add positional bonus
                    if piece['type'] in PST:
                        pst = PST[piece['type']]
                        if piece['color'] == 'w':
                            val += pst[r][c]
                        else:
                            val += pst[7-r][c]
                    
                    if piece['color'] == 'w':
                        score += val
                    else:
                        score -= val
        
        # Return from perspective of current player
        return score if self.engine.turn == 'w' else -score

    def get_opening_move(self):
        history = ''.join(self.engine.coordinate_history)
        
        if history in OPENING_BOOK:
            responses = OPENING_BOOK[history]
            move_str = random.choice(responses)
            move = self.engine.parse_move(move_str)
            if move:
                # Verify move is legal
                legal_moves = self.engine.get_all_legal_moves(self.engine.turn)
                for lm in legal_moves:
                    if lm['from'] == move['from'] and lm['to'] == move['to']:
                        print(f"  [AI] Book Move: {move_str}")
                        return lm
        return None
