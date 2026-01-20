import random
import time
"""
Chess AI with Supervised Learning
Cờ Vua AI học có giám sát
"""

import json
import os
from config import *

# Piece values
SIMPLE_VALUES = {'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000}

CENTER = [(3, 3), (3, 4), (4, 3), (4, 4)]
EXTENDED_CENTER = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)]
CORNER_SQUARES = [(0, 0), (0, 7), (7, 0), (7, 7)]


class ChessAI:
    def __init__(self, engine):
        self.engine = engine
        self.difficulty = 'medium'
        self.max_time = 3.0
        
        # Learning system
        self.learning_file = "chess_ai_learning.json"
        self.experience = self._load_experience()
        self.game_moves = []  # Track moves for this game
        self.start_time = None

    def _load_experience(self):
        """Load previously learned patterns"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'games_played': 0, 'move_patterns': {}, 'position_evals': {}}

    def _save_experience(self):
        """Save learned patterns to disk"""
        try:
            with open(self.learning_file, 'w') as f:
                json.dump(self.experience, f, indent=2)
        except:
            pass

    def record_game_move(self, move, move_index):
        """Record a move made during the game"""
        move_key = f"{move['from']}->{move['to']}"
        self.game_moves.append({
            'move_key': move_key,
            'move': move,
            'index': move_index
        })

    def analyze_game(self, ai_color, result):
        """
        Analyze completed game for supervised learning
        result: 'win', 'loss', 'draw'
        """
        self.experience['games_played'] += 1
        
        # Convert result to label
        result_label = {'win': 1, 'loss': -1, 'draw': 0}.get(result, 0)
        
        # Record move outcomes based on game result
        for game_move in self.game_moves:
            move_key = game_move['move_key']
            
            if move_key not in self.experience['move_patterns']:
                self.experience['move_patterns'][move_key] = {
                    'wins': 0, 'losses': 0, 'draws': 0, 'count': 0
                }
            
            stats = self.experience['move_patterns'][move_key]
            stats['count'] += 1
            
            if result_label == 1:
                stats['wins'] += 1
            elif result_label == -1:
                stats['losses'] += 1
            else:
                stats['draws'] += 1
        
        self._save_experience()
        self.game_moves = []
        
        print(f"[AI Learning] Game {self.experience['games_played']} - Result: {result}")

    def _get_piece_value(self, piece):
        """Get material value of a piece"""
        return SIMPLE_VALUES.get(piece['type'], 0) if piece else 0

    def _is_opening_phase(self):
        """Check if we're in opening phase"""
        return len(self.engine.move_history) < 20

    def _is_endgame_phase(self):
        """Check if we're in endgame (no queens)"""
        queens = sum(1 for r in range(8) for c in range(8) 
                    if self.engine.board[r][c] and self.engine.board[r][c]['type'] == 'q')
        return queens == 0

    def _count_attacks_on_square(self, r, c, color):
        """Count how many pieces of 'color' attack square (r,c)"""
        count = 0
        for row in range(8):
            for col in range(8):
                p = self.engine.board[row][col]
                if p and p['color'] == color:
                    moves = self.engine.get_piece_moves(row, col, check_legal=False)
                    for m in moves:
                        if m['to'] == (r, c):
                            count += 1
                            break
        return count

    def _is_losing_move(self, move):
        """Check if this move loses material"""
        fr, fc = move['from']
        tr, tc = move['to']
        
        piece = self.engine.board[fr][fc]
        if not piece:
            return False
        
        my_color = piece['color']
        opp_color = 'b' if my_color == 'w' else 'w'
        my_value = self._get_piece_value(piece)
        
        target = self.engine.board[tr][tc]
        capture_value = self._get_piece_value(target)
        
        # Simulate move to check if piece is attacked
        saved_from, saved_to = self.engine.board[fr][fc], self.engine.board[tr][tc]
        self.engine.board[tr][tc] = saved_from
        self.engine.board[fr][fc] = None
        
        is_attacked = self.engine.is_square_attacked(tr, tc, opp_color)
        is_defended = self.engine.is_square_attacked(tr, tc, my_color)
        
        # Restore board
        self.engine.board[fr][fc] = saved_from
        self.engine.board[tr][tc] = saved_to
        
        if is_attacked:
            if not is_defended:
                material_loss = my_value - capture_value
                if material_loss > 0:
                    return True
            else:
                if capture_value < my_value - 30:
                    return True
        return False

    def _evaluate_move_basic(self, move):
        """
        Basic move evaluation:
        - Captures
        - Threats
        - Defense
        - Piece development (opening)
        """
        fr, fc = move['from']
        tr, tc = move['to']
        
        piece = self.engine.board[fr][fc]
        if not piece:
            return 0
        
        score = 0
        my_color = piece['color']
        opp_color = 'b' if my_color == 'w' else 'w'
        
        # 1. Capture scoring
        target = self.engine.board[tr][tc]
        if target:
            score += self._get_piece_value(target) * 10
        
        # 2. Center control
        if (tr, tc) in CENTER:
            score += 30
        elif (tr, tc) in EXTENDED_CENTER:
            score += 10
        
        # 3. Piece safety after move
        saved_from, saved_to = self.engine.board[fr][fc], self.engine.board[tr][tc]
        self.engine.board[tr][tc] = saved_from
        self.engine.board[fr][fc] = None
        
        is_attacked = self.engine.is_square_attacked(tr, tc, opp_color)
        is_defended = self.engine.is_square_attacked(tr, tc, my_color)
        
        self.engine.board[fr][fc] = saved_from
        self.engine.board[tr][tc] = saved_to
        
        if is_attacked and not is_defended:
            score -= self._get_piece_value(piece)
        elif is_attacked:
            score -= self._get_piece_value(piece) // 4
        
        # 4. Development in opening
        if self._is_opening_phase():
            back_rank = 7 if my_color == 'w' else 0
            
            # Develop minor pieces
            if piece['type'] in ['n', 'b'] and fr == back_rank:
                score += 150
            
            # Center pawns
            if piece['type'] == 'p' and (tr, tc) in CENTER:
                score += 50
            
            # Castling
            if move.get('castling'):
                score += 200
        
        # 5. Check if opponent
        king_pos = self.engine.find_king(opp_color)
        if king_pos:
            kr, kc = king_pos
            piece_moves = self.engine.get_piece_moves(tr, tc, check_legal=False)
            for m in piece_moves:
                if m['to'] == king_pos:
                    score += 100  # Checks are good
        
        return score

    def _get_move_quality_from_learning(self, move):
        """Get quality score of a move based on historical data"""
        move_key = f"{move['from']}->{move['to']}"
        
        if move_key not in self.experience['move_patterns']:
            return 0.0
        
        stats = self.experience['move_patterns'][move_key]
        if stats['count'] == 0:
            return 0.0
        
        # Win rate of this move
        win_rate = (stats['wins'] - stats['losses']) / stats['count']
        return win_rate

    def get_best_move(self):
        """
        Get the best move using:
        1. Captures and threats (tactical)
        2. Historical learning (supervised)
        3. Opening book
        """
        if not self.engine.game_active:
            return None
        
        # Try opening book first
        book_move = self.get_opening_move()
        if book_move:
            self.record_game_move(book_move, len(self.engine.move_history))
            return book_move
        
        all_moves = self.engine.get_all_legal_moves(self.engine.turn)
        if not all_moves:
            return None
        
        my_color = self.engine.turn
        
        # Filter out losing moves
        safe_moves = [m for m in all_moves if not self._is_losing_move(m)]
        moves_to_search = safe_moves if safe_moves else all_moves
        
        # Score each move
        best_move = None
        best_score = -float('inf')
        
        self.start_time = time.time()
        
        for move in moves_to_search:
            if time.time() - self.start_time > self.max_time:
                break
            
            # Combine tactical evaluation + learning
            tactical_score = self._evaluate_move_basic(move)
            learning_score = self._get_move_quality_from_learning(move) * 100
            
            total_score = tactical_score + learning_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = move
        
        if best_move:
            self.record_game_move(best_move, len(self.engine.move_history))
        
        return best_move

    def get_opening_move(self):
        """Get move from opening book if available"""
        history = ' '.join(self.engine.coordinate_history)
        
        if history in OPENING_BOOK:
            choices = OPENING_BOOK[history]
            if choices:
                move_str = random.choice(choices)
                move = self.engine.parse_move(move_str)
                if move:
                    legal = self.engine.get_all_legal_moves(self.engine.turn)
                    for lm in legal:
                        if lm['from'] == move['from'] and lm['to'] == move['to']:
                            return lm
        return None
