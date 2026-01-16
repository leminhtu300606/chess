import random
import time
import json
import os
from config import *

# Piece values
SIMPLE_VALUES = {'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000}

# Strategic squares
CENTER = [(3, 3), (3, 4), (4, 3), (4, 4)]
EXTENDED_CENTER = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)]

class ChessAI:
    def __init__(self, engine):
        self.engine = engine
        self.difficulty = 'medium'
        self.max_time = 3.0
        
        self.learning_file = "chess_ai_learning.json"
        self.experience = self._load_experience()
        self.game_positions = []

    def _load_experience(self):
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'bad_moves': {}, 'games_played': 0}

    def _save_experience(self):
        try:
            with open(self.learning_file, 'w') as f:
                json.dump(self.experience, f, indent=2)
        except:
            pass

    def record_position(self, move, ai_color):
        self.game_positions.append({'move': str(move), 'material': self._count_material()})

    def _count_material(self):
        balance = 0
        for r in range(8):
            for c in range(8):
                p = self.engine.board[r][c]
                if p and p['type'] != 'k':
                    v = SIMPLE_VALUES.get(p['type'], 0)
                    balance += v if p['color'] == 'w' else -v
        return balance

    def analyze_game(self, ai_color, result):
        self.experience['games_played'] += 1
        self._save_experience()
        self.game_positions = []

    def _get_piece_value(self, piece):
        return SIMPLE_VALUES.get(piece['type'], 0) if piece else 0

    def _is_opening_phase(self):
        return len(self.engine.move_history) < 20

    def _is_endgame_phase(self):
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
        
        # Simulate move
        saved_from, saved_to = self.engine.board[fr][fc], self.engine.board[tr][tc]
        self.engine.board[tr][tc] = saved_from
        self.engine.board[fr][fc] = None
        
        is_attacked = self.engine.is_square_attacked(tr, tc, opp_color)
        is_defended = self.engine.is_square_attacked(tr, tc, my_color)
        
        self.engine.board[fr][fc] = saved_from
        self.engine.board[tr][tc] = saved_to
        
        if is_attacked:
            if not is_defended:
                # Undefended piece: SEVERE penalty for losing material
                material_loss = my_value - capture_value
                if material_loss > 0:
                    return True  # Any material loss is bad!
            else:
                # Defended piece (trade)
                # Only allow if we're getting good value
                if capture_value < my_value - 30:  # Stricter threshold
                    return True  # Bad trade!
        return False

    def _evaluate_tactical_move(self, move):
        """Evaluate tactical value of a move"""
        fr, fc = move['from']
        tr, tc = move['to']
        
        piece = self.engine.board[fr][fc]
        if not piece:
            return 0
        
        my_color = piece['color']
        opp_color = 'b' if my_color == 'w' else 'w'
        score = 0
        
        # Capture value
        target = self.engine.board[tr][tc]
        if target:
            score += self._get_piece_value(target)
        
        # === TACTICAL BONUSES ===
        
        # 1. DOUBLE ATTACK potential (fork check)
        # After this move, count how many opponent pieces we attack
        saved_from, saved_to = self.engine.board[fr][fc], self.engine.board[tr][tc]
        self.engine.board[tr][tc] = saved_from
        self.engine.board[fr][fc] = None
        
        attacks_after = 0
        valuable_attacks = 0
        moves_after = self.engine.get_piece_moves(tr, tc, check_legal=False)
        for m in moves_after:
            target_sq = self.engine.board[m['to'][0]][m['to'][1]]
            if target_sq and target_sq['color'] == opp_color:
                attacks_after += 1
                if target_sq['type'] in ['q', 'r', 'k']:
                    valuable_attacks += 1
        
        # Double attack bonus (attacking 2+ pieces)
        if attacks_after >= 2:
            score += 30 + valuable_attacks * 20
        
        # 2. ROOK ON 7TH RANK (very powerful!)
        if piece['type'] == 'r':
            seventh_rank = 1 if my_color == 'w' else 6
            if tr == seventh_rank:
                score += 50  # Rook on 7th rank is very strong
        
        # 3. QUEEN + ROOK coordination check
        # If we have queen and rook on same file/rank
        if piece['type'] in ['q', 'r']:
            for row in range(8):
                for col in range(8):
                    other = self.engine.board[row][col]
                    if other and other['color'] == my_color:
                        if other['type'] == 'q' and piece['type'] == 'r':
                            if row == tr or col == tc:
                                score += 25  # Coordination bonus
                        elif other['type'] == 'r' and piece['type'] == 'q':
                            if row == tr or col == tc:
                                score += 25
        
        # === NEW: PIECE BLOCKING PENALTY ===
        # Check if this move blocks friendly long-range pieces (Bishop, Rook, Queen)
        blocking_penalty = 0
        for row in range(8):
            for col in range(8):
                fp = self.engine.board[row][col]
                if fp and fp['color'] == my_color and fp['type'] in ['b', 'q', 'r'] and (row, col) != (fr, fc):
                    # Check if (tr, tc) blocks the line of sight for this piece
                    dr = tr - row
                    dc = tc - col
                    
                    # Check if they are on same diagonal, rank or file
                    is_diag = abs(dr) == abs(dc) and dr != 0
                    is_line = (dr == 0 or dc == 0) and (dr != 0 or dc != 0)
                    
                    if (fp['type'] in ['b', 'q'] and is_diag) or (fp['type'] in ['r', 'q'] and is_line):
                        # The move is on a line this piece could use
                        # Penalty is higher if we're closer to the piece (blocking more potential)
                        dist = max(abs(dr), abs(dc))
                        # SEVERE PENALTY: blocks friendly piece's development
                        # Penalty ranges from -150 (dist 1) to -50 (dist 4)
                        blocking_penalty -= max(50, 180 - dist * 30)
                        
                        # Extra penalty for blocking a piece still on its home square
                        home_row = 7 if my_color == 'w' else 0
                        if row == home_row:
                            blocking_penalty -= 50
                            
        score += blocking_penalty
        
        # === NEW: ESCAPE ROUTE CHECK ===
        # After moving, does the piece have safe retreats?
        escape_squares = 0
        for m in moves_after:
            escape_r, escape_c = m['to']
            if not self.engine.is_square_attacked(escape_r, escape_c, opp_color):
                escape_squares += 1
        if piece['type'] in ['n', 'b'] and escape_squares == 0:
            score -= 50  # No escape routes = dangerous position
        
        # === NEW: ROOK TIMING ===
        # Don't move rooks before castling!
        if piece['type'] == 'r' and self._is_opening_phase():
            if self.engine.castling_rights[my_color]['k'] or self.engine.castling_rights[my_color]['q']:
                score -= 100  # Keep rooks for castling!
        
        # === NEW: THREAT RESPONSE PRIORITY ===
        # If a piece is currently threatened, prioritize moves that save it
        if self.engine.is_square_attacked(fr, fc, opp_color):
            my_val = self._get_piece_value(piece)
            # Check what's attacking us
            for ar in range(8):
                for ac in range(8):
                    attacker = self.engine.board[ar][ac]
                    if attacker and attacker['color'] == opp_color:
                        attacker_moves = self.engine.get_piece_moves(ar, ac, check_legal=False)
                        for am in attacker_moves:
                            if am['to'] == (fr, fc):
                                attacker_val = self._get_piece_value(attacker)
                                # Being attacked by lower value piece = MUST FLEE
                                if attacker_val < my_val:
                                    # This move saves the piece!
                                    if not self.engine.is_square_attacked(tr, tc, opp_color):
                                        score += my_val  # Big bonus for escaping
                                    else:
                                        score -= my_val // 2  # Still in danger
        
        # 4. KNIGHT FORK potential
        if piece['type'] == 'n':
            high_value_attacks = 0
            for m in moves_after:
                target_sq = self.engine.board[m['to'][0]][m['to'][1]]
                if target_sq and target_sq['color'] == opp_color:
                    if target_sq['type'] in ['q', 'r', 'k']:
                        high_value_attacks += 1
            if high_value_attacks >= 2:
                score += 100
        
        # 5. Check if move is safe (MUST BE BEFORE pawn interception that uses is_defended)
        is_attacked = self.engine.is_square_attacked(tr, tc, opp_color)
        is_defended = self.engine.is_square_attacked(tr, tc, my_color)
        
        # Restore board state
        self.engine.board[fr][fc] = saved_from
        self.engine.board[tr][tc] = saved_to
        
        # 6. PAWN INTERCEPTION / ATTACK
        if piece['type'] == 'p':
            pawn_dir = -1 if my_color == 'w' else 1
            for dc in [-1, 1]:
                attack_r, attack_c = tr + pawn_dir, tc + dc
                if 0 <= attack_r < 8 and 0 <= attack_c < 8:
                    target_sq = self.engine.board[attack_r][attack_c]
                    if target_sq and target_sq['color'] == opp_color:
                        if target_sq['type'] in ['q', 'r', 'b']:
                            score += 40
                            if is_defended:
                                score += 20
        
        if is_attacked and not is_defended:
            score -= self._get_piece_value(piece)
        elif is_attacked:
            score -= self._get_piece_value(piece) // 4
        
        # 6. Opening principles - STRICT!
        if self._is_opening_phase():
            back_rank = 7 if my_color == 'w' else 0
            
            # HEAVY penalty for early queen
            if piece['type'] == 'q':
                move_count = len(self.engine.move_history)
                if move_count < 10:
                    score -= 300  # NEVER move queen this early
                elif move_count < 20:
                    score -= 150
            
            if piece['type'] == 'k' and not move.get('castling'):
                score -= 250  # Never move king without castling in opening
            if move.get('castling'):
                score += 200  # Castling is excellent
            
            # HUGE bonus for developing minor pieces (Knights and Bishops)
            if piece['type'] in ['n', 'b'] and fr == back_rank:
                score += 200  # VERY high priority for development!
                # Extra bonus for developing towards center
                if 2 <= tc <= 5:  # Developing to central files
                    score += 50
                # Knights to f3/c3/f6/c6 are ideal
                if piece['type'] == 'n':
                    if (tr, tc) in [(5, 2), (5, 5), (2, 2), (2, 5)]:  # c3, f3, c6, f6
                        score += 40
            
            # Pawn moves - only encourage center pawns
            if piece['type'] == 'p':
                if (tr, tc) in CENTER:
                    score += 60  # Control center with pawns is good
                else:
                    # Penalty for non-central pawn moves if pieces not developed
                    undeveloped_pieces = 0
                    for check_c in range(8):
                        p = self.engine.board[back_rank][check_c]
                        if p and p['type'] in ['n', 'b'] and p['color'] == my_color:
                            undeveloped_pieces += 1
                    if undeveloped_pieces >= 2:
                        score -= 30  # Don't push side pawns before developing
        
        # 7. Center control - Bonus for Occupying OR Attacking center
        if (tr, tc) in CENTER:
            score += 30
        elif (tr, tc) in EXTENDED_CENTER:
            score += 15
        
        # 8. CAPTURE BONUS - Very important!
        if target:
            capture_val = self._get_piece_value(target)
            score += capture_val + 100  # Increased base bonus from 50 to 100
            
            # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            my_val = self._get_piece_value(piece)
            if capture_val >= my_val:
                score += 150  # Increased from 100 - Great trade or win material
            elif capture_val >= my_val - 100:
                score += 80   # Increased from 50 - Fair trade
            
            # === NEW: STRATEGIC EXCHANGE BONUS ===
            # Bonus for removing opponent's developed minor pieces
            if target['type'] in ['n', 'b']:
                opp_back_rank = 0 if my_color == 'w' else 7
                target_r, target_c = tr, tc
                if target_r != opp_back_rank:
                    score += 60  # Removing developed minor piece is great
                
                # Special bonus for same-color square capture (often Bishop vs Knight)
                # Helps AI recognize the value of the specific exchange shown by user
                if (fr + fc) % 2 == (target_r + target_c) % 2:
                    score += 40  # Same-color square exchange bonus
                    
            # Bonus for removing the queen
            if target['type'] == 'q':
                score += 200  # Removing opponent queen is a top priority
        
        # 9. REPETITION PENALTY - Don't move back to where we came from!
        if len(self.engine.move_history) >= 2:
            last_move_str = self.engine.move_history[-1] if self.engine.move_history else ""
            # Check if we're moving back to origin of last move
            if last_move_str:
                # Last move was FROM some square TO current piece position
                # If we go back there, it's repetition
                try:
                    cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
                    last_from_c = cols.get(last_move_str[0], -1)
                    last_from_r = 8 - int(last_move_str[1]) if len(last_move_str) > 1 else -1
                    if (tr, tc) == (last_from_r, last_from_c):
                        score -= 80  # Heavy penalty for moving back!
                except:
                    pass
        
        # 10. PAWN BLOCKING - Use pawns to block advancing pieces
        if piece['type'] == 'p':
            # Check if this pawn move blocks an enemy piece's advance
            enemy_dir = 1 if my_color == 'w' else -1
            blocking_row = tr + enemy_dir
            if 0 <= blocking_row < 8:
                # Check if enemy piece was threatening to advance here
                blocker_target = self.engine.board[blocking_row][tc]
                if not blocker_target:  # Square is empty, check if enemy could use it
                    # Check if enemy has pieces that could reach blocking_row, tc
                    if self.engine.is_square_attacked(blocking_row, tc, opp_color):
                        score += 25  # Good blocking move
            
            # Pawn chain support
            for dc in [-1, 1]:
                support_c = tc + dc
                pawn_behind_r = tr - enemy_dir
                if 0 <= support_c < 8 and 0 <= pawn_behind_r < 8:
                    support_piece = self.engine.board[pawn_behind_r][support_c]
                    if support_piece and support_piece['type'] == 'p' and support_piece['color'] == my_color:
                        score += 15  # Supported by another pawn
        
        # === 11. PROTECTION BONUS - Reward moves that defend valuable pieces ===
        # After moving, check if we now defend any valuable friendly pieces
        saved_from, saved_to = self.engine.board[fr][fc], self.engine.board[tr][tc]
        self.engine.board[tr][tc] = piece
        self.engine.board[fr][fc] = None
        
        protection_bonus = 0
        # Get what squares we now attack
        our_attacks = self.engine.get_piece_moves(tr, tc, check_legal=False)
        for atk in our_attacks:
            atk_r, atk_c = atk['to']
            defended_piece = self.engine.board[atk_r][atk_c]
            if defended_piece and defended_piece['color'] == my_color:
                defended_val = self._get_piece_value(defended_piece)
                my_val = self._get_piece_value(piece)
                if defended_val > my_val:
                    # We're defending a more valuable piece!
                    protection_bonus += defended_val // 10
                # Extra bonus for defending king
                if defended_piece['type'] == 'k':
                    protection_bonus += 50
        
        # Find king position and check if we're defending it
        king_pos = None
        for kr in range(8):
            for kc in range(8):
                kp = self.engine.board[kr][kc]
                if kp and kp['type'] == 'k' and kp['color'] == my_color:
                    king_pos = (kr, kc)
                    break
        
        # Bonus for being near king (defensive position)
        if king_pos and piece['type'] in ['n', 'b', 'p']:
            dist = abs(tr - king_pos[0]) + abs(tc - king_pos[1])
            if dist <= 2:
                protection_bonus += 20  # Close to king
        
        self.engine.board[fr][fc] = saved_from
        self.engine.board[tr][tc] = saved_to
        
        score += protection_bonus
        
        # === 12. PURPOSE CHECK - Penalize moves with no clear purpose ===
        has_purpose = False
        
        # Purpose 1: Capturing
        if target:
            has_purpose = True
        
        # Purpose 2: Developing from back rank
        back_rank = 7 if my_color == 'w' else 0
        if fr == back_rank and piece['type'] in ['n', 'b']:
            has_purpose = True
        
        # Purpose 3: Moving to or attacking center
        if (tr, tc) in CENTER or (tr, tc) in EXTENDED_CENTER:
            has_purpose = True
        
        # Purpose 4: Defending a piece under attack
        for dr in range(8):
            for dc in range(8):
                fp = self.engine.board[dr][dc]
                if fp and fp['color'] == my_color and fp['type'] != 'k':
                    if self.engine.is_square_attacked(dr, dc, opp_color):
                        # Check if our move adds defense
                        for atk in our_attacks if 'our_attacks' in dir() else []:
                            if atk['to'] == (dr, dc):
                                has_purpose = True
        
        # Purpose 5: Castling
        if move.get('castling'):
            has_purpose = True
        
        # Purpose 6: Creating threats
        if attacks_after > 0:
            has_purpose = True
        
        # Purpose 7: Pawn push towards promotion
        if piece['type'] == 'p':
            promo_rank = 0 if my_color == 'w' else 7
            if abs(tr - promo_rank) <= 3:
                has_purpose = True
        
        # No purpose = heavy penalty
        if not has_purpose:
            score -= 100  # Pointless move!
        
        return score

    def get_best_move(self):
        if not self.engine.game_active:
            return None
        
        book_move = self.get_opening_move()
        if book_move:
            return book_move
        
        depth = {'easy': 2, 'medium': 3, 'hard': 4}.get(self.difficulty, 3)
        
        all_moves = self.engine.get_all_legal_moves(self.engine.turn)
        if not all_moves:
            return None
        
        my_color = self.engine.turn
        opp_color = 'b' if my_color == 'w' else 'w'
        
        # === PRIORITY 1: Check if we have a free capture ===
        capture_moves = []
        for m in all_moves:
            tr, tc = m['to']
            target = self.engine.board[tr][tc]
            if target:
                fr, fc = m['from']
                attacker = self.engine.board[fr][fc]
                if attacker:
                    target_val = self._get_piece_value(target)
                    attacker_val = self._get_piece_value(attacker)
                    
                    # Simulate the capture
                    self.engine.make_move(m, switch_turn=True, record_history=True)
                    # Check if our piece is safe after capture
                    is_safe = not self.engine.is_square_attacked(tr, tc, opp_color)
                    self.engine.undo_move()
                    
                    if is_safe:
                        # Free capture! 
                        capture_moves.append((m, target_val))
                    elif target_val >= attacker_val:
                        # Good trade
                        capture_moves.append((m, target_val - attacker_val + 50))
        
        # Sort captures by value and take best one if it's good
        if capture_moves:
            capture_moves.sort(key=lambda x: x[1], reverse=True)
            best_capture, best_val = capture_moves[0]
            if best_val >= 100:  # At least a pawn's worth
                return best_capture
        
        # === PRIORITY 2: Check if any of our pieces need to flee ===
        for r in range(8):
            for c in range(8):
                piece = self.engine.board[r][c]
                if piece and piece['color'] == my_color and piece['type'] != 'k':
                    if self.engine.is_square_attacked(r, c, opp_color):
                        # This piece is under attack - find escape moves
                        my_val = self._get_piece_value(piece)
                        escape_moves = []
                        for m in all_moves:
                            if m['from'] == (r, c):
                                tr, tc = m['to']
                                # Check if destination is safe
                                if not self.engine.is_square_attacked(tr, tc, opp_color):
                                    escape_moves.append(m)
                        if escape_moves:
                            # Return best escape (prefer one that also attacks)
                            escape_moves.sort(key=lambda m: self._evaluate_tactical_move(m), reverse=True)
                            return escape_moves[0]
        
        # === PRIORITY 3: Normal search with lookahead ===
        safe_moves = [m for m in all_moves if not self._is_losing_move(m)]
        moves_to_search = safe_moves if safe_moves else all_moves
        
        # Better move ordering: captures first, then tactical moves
        def move_priority(m):
            tr, tc = m['to']
            target = self.engine.board[tr][tc]
            priority = self._evaluate_tactical_move(m)
            if target:
                priority += 1000  # Captures get huge priority in ordering
            return priority
        
        moves_to_search.sort(key=move_priority, reverse=True)
        
        best_move = moves_to_search[0]
        best_value = -float('inf')
        alpha, beta = -float('inf'), float('inf')
        
        self.start_time = time.time()
        
        for move in moves_to_search:
            if time.time() - self.start_time > self.max_time:
                break
            
            # === LOOKAHEAD: Simulate opponent's response ===
            self.engine.make_move(move, switch_turn=True, record_history=True)
            
            # Get opponent's best response
            opp_moves = self.engine.get_all_legal_moves(self.engine.turn)
            opp_threat = 0
            for om in opp_moves[:5]:  # Check top 5 opponent moves
                otr, otc = om['to']
                our_piece = self.engine.board[otr][otc]
                if our_piece and our_piece['color'] == my_color:
                    opp_threat = max(opp_threat, self._get_piece_value(our_piece))
            
            val = self.minimax(depth - 1, alpha, beta, False)
            val -= opp_threat // 3  # Penalize moves that allow opponent threats
            
            self.engine.undo_move()
            
            val += self._evaluate_tactical_move(move) // 2
            
            # Extra penalty for repetition
            if self._is_repetitive_move(move):
                val -= 200
            
            if val > best_value:
                best_value = val
                best_move = move
            alpha = max(alpha, best_value)
        
        # Final safety check
        if self._is_losing_move(best_move) and safe_moves:
            best_move = safe_moves[0]
        
        return best_move
    
    def _is_repetitive_move(self, move):
        """Check if this move creates a repetitive pattern"""
        if len(self.engine.move_history) < 4:
            return False
        
        fr, fc = move['from']
        tr, tc = move['to']
        
        # Check if we're going back to a square we left recently
        for i in range(min(4, len(self.engine.move_history))):
            hist = self.engine.move_history[-(i+1)]
            try:
                cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
                h_fc = cols.get(hist[0], -1)
                h_fr = 8 - int(hist[1])
                h_tc = cols.get(hist[2], -1)
                h_tr = 8 - int(hist[3])
                
                # If we're moving to where a recent move came from
                if (tr, tc) == (h_fr, h_fc):
                    return True
                # If we're moving from where a recent move went to
                if (fr, fc) == (h_tr, h_tc) and (tr, tc) == (h_fr, h_fc):
                    return True
            except:
                pass
        return False

    def minimax(self, depth, alpha, beta, is_max):
        if time.time() - self.start_time > self.max_time:
            return self.evaluate()
        
        if depth == 0:
            return self.quiescence(alpha, beta, is_max, 4)

        moves = self.engine.get_all_legal_moves(self.engine.turn)
        
        if not moves:
            if self.engine.is_in_check(self.engine.turn):
                return -50000 + depth if is_max else 50000 - depth
            return 0
        
        moves.sort(key=lambda m: self._evaluate_tactical_move(m), reverse=is_max)
        
        if is_max:
            max_eval = -float('inf')
            for move in moves[:12]:
                self.engine.make_move(move, switch_turn=True, record_history=True)
                max_eval = max(max_eval, self.minimax(depth - 1, alpha, beta, False))
                self.engine.undo_move()
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves[:12]:
                self.engine.make_move(move, switch_turn=True, record_history=True)
                min_eval = min(min_eval, self.minimax(depth - 1, alpha, beta, True))
                self.engine.undo_move()
                beta = min(beta, min_eval)
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
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        moves = self.engine.get_all_legal_moves(self.engine.turn)
        captures = []
        for m in moves:
            tr, tc = m['to']
            target = self.engine.board[tr][tc]
            if target:
                fr, fc = m['from']
                attacker = self.engine.board[fr][fc]
                target_val = self._get_piece_value(target)
                attacker_val = self._get_piece_value(attacker)
                
                # Capture Score: MVV-LVA
                # Always check captures unless it's obviously bad (Queen takes protected Pawn)
                # But here we want to see even "bad" captures if they open lines, so be lenient
                score_val = target_val * 10 - attacker_val
                captures.append((m, score_val))
                
        captures.sort(key=lambda x: x[1], reverse=True)
        
        # Consider top captures
        for m, _ in captures[:8]:
            self.engine.make_move(m, switch_turn=True, record_history=True)
            score = self.quiescence(alpha, beta, not is_max, depth - 1)
            self.engine.undo_move()

            if is_max:
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            else:
                if score <= alpha:
                    return alpha
                beta = min(beta, score)

        return alpha if is_max else beta

    def evaluate(self):
        """Full position evaluation with advanced tactical awareness"""
        score = 0
        w_bishops, b_bishops = 0, 0
        w_center, b_center = 0, 0
        w_mobility, b_mobility = 0, 0
        w_space, b_space = 0, 0
        w_developed, b_developed = 0, 0
        
        # Track pawn files for structure analysis
        w_pawn_files = [0] * 8
        b_pawn_files = [0] * 8
        
        for r in range(8):
            for c in range(8):
                piece = self.engine.board[r][c]
                if not piece:
                    continue
                
                base_val = SIMPLE_VALUES.get(piece['type'], 0)
                
                # PST bonus
                pos_val = 0
                if piece['type'] in PST:
                    row = r if piece['color'] == 'w' else 7 - r
                    pos_val = PST[piece['type']][row][c]
                
                piece_score = base_val + pos_val
                
                # === 1. CENTER CONTROL ===
                if (r, c) in CENTER:
                    piece_score += 35
                    if piece['color'] == 'w':
                        w_center += 1
                    else:
                        b_center += 1
                elif (r, c) in EXTENDED_CENTER:
                    piece_score += 15
                
                # === 2. PIECE MOBILITY ===
                if piece['type'] in ['n', 'b', 'r', 'q']:
                    moves = self.engine.get_piece_moves(r, c, check_legal=False)
                    mobility = len(moves)
                    mobility_bonus = mobility * 3
                    piece_score += mobility_bonus
                    if piece['color'] == 'w':
                        w_mobility += mobility
                    else:
                        b_mobility += mobility
                
                # === 3. PIECE DEVELOPMENT ===
                back_rank = 7 if piece['color'] == 'w' else 0
                if piece['type'] in ['n', 'b']:
                    if r != back_rank:
                        piece_score += 40  # Increased from 25
                        if piece['color'] == 'w':
                            w_developed += 1
                        else:
                            b_developed += 1
                        
                        # === CENTRAL PIECE PROTECTION ===
                        # Pieces in center that are defended get bonus
                        if (r, c) in CENTER or (r, c) in EXTENDED_CENTER:
                            opp = 'b' if piece['color'] == 'w' else 'w'
                            is_attacked = self.engine.is_square_attacked(r, c, opp)
                            is_defended = self.engine.is_square_attacked(r, c, piece['color'])
                            if is_defended:
                                piece_score += 30  # Protected central piece = excellent
                            if is_attacked and not is_defended:
                                piece_score -= 50  # Unprotected in center = very bad
                
                # === 4. BISHOP PAIR ===
                if piece['type'] == 'b':
                    if piece['color'] == 'w':
                        w_bishops += 1
                    else:
                        b_bishops += 1
                
                # === 5. ROOK EVALUATION ===
                if piece['type'] == 'r':
                    seventh = 1 if piece['color'] == 'w' else 6
                    if r == seventh:
                        piece_score += 50  # Rook on 7th rank
                    
                    # Open file bonus
                    has_own_pawn = any(self.engine.board[row][c] and 
                                      self.engine.board[row][c]['type'] == 'p' and
                                      self.engine.board[row][c]['color'] == piece['color']
                                      for row in range(8))
                    has_opp_pawn = any(self.engine.board[row][c] and 
                                      self.engine.board[row][c]['type'] == 'p' and
                                      self.engine.board[row][c]['color'] != piece['color']
                                      for row in range(8))
                    if not has_own_pawn and not has_opp_pawn:
                        piece_score += 40  # Open file
                    elif not has_own_pawn:
                        piece_score += 20  # Semi-open file
                
                # === 6. PAWN STRUCTURE ===
                if piece['type'] == 'p':
                    if piece['color'] == 'w':
                        w_pawn_files[c] += 1
                        # Passed pawn bonus
                        is_passed = True
                        for check_r in range(r - 1, -1, -1):
                            for check_c in [c - 1, c, c + 1]:
                                if 0 <= check_c < 8:
                                    p = self.engine.board[check_r][check_c]
                                    if p and p['type'] == 'p' and p['color'] == 'b':
                                        is_passed = False
                        if is_passed:
                            piece_score += 30 + (7 - r) * 10  # More valuable closer to promotion
                    else:
                        b_pawn_files[c] += 1
                        is_passed = True
                        for check_r in range(r + 1, 8):
                            for check_c in [c - 1, c, c + 1]:
                                if 0 <= check_c < 8:
                                    p = self.engine.board[check_r][check_c]
                                    if p and p['type'] == 'p' and p['color'] == 'w':
                                        is_passed = False
                        if is_passed:
                            piece_score += 30 + r * 10
                
                # === 7. HANGING PIECE PENALTY ===
                if piece['type'] != 'k':
                    opp = 'b' if piece['color'] == 'w' else 'w'
                    if self.engine.is_square_attacked(r, c, opp):
                        if not self.engine.is_square_attacked(r, c, piece['color']):
                            piece_score -= base_val * 2  # Hanging!
                        else:
                            piece_score -= base_val // 4
                
                # === 8. KING SAFETY ===
                if piece['type'] == 'k':
                    if not self._is_endgame_phase():
                        # Castled king bonus
                        if c in [0, 1, 2, 6, 7]:
                            piece_score += 40
                        if r in [0, 7]:
                            piece_score += 30
                        # Pawn shield
                        pawn_shield = 0
                        shield_row = r - 1 if piece['color'] == 'w' else r + 1
                        if 0 <= shield_row < 8:
                            for dc in [-1, 0, 1]:
                                sc = c + dc
                                if 0 <= sc < 8:
                                    sp = self.engine.board[shield_row][sc]
                                    if sp and sp['type'] == 'p' and sp['color'] == piece['color']:
                                        pawn_shield += 15
                        piece_score += pawn_shield
                    else:
                        # In endgame, king should be active
                        if (r, c) in CENTER or (r, c) in EXTENDED_CENTER:
                            piece_score += 20
                
                # === 9. SPACE CONTROL ===
                if piece['color'] == 'w':
                    if r < 4:  # In opponent's half
                        w_space += 1
                else:
                    if r > 3:
                        b_space += 1
                
                # Add piece score
                if piece['color'] == 'w':
                    score += piece_score
                else:
                    score -= piece_score
        
        # === GLOBAL BONUSES ===
        
        # Bishop pair
        if w_bishops >= 2:
            score += 50
        if b_bishops >= 2:
            score -= 50
        
        # Center control advantage
        center_diff = w_center - b_center
        score += center_diff * 40
        
        # Mobility advantage
        mobility_diff = w_mobility - b_mobility
        score += mobility_diff * 2
        
        # Space advantage
        space_diff = w_space - b_space
        score += space_diff * 5
        
        # Development advantage (especially in opening)
        if self._is_opening_phase():
            dev_diff = w_developed - b_developed
            score += dev_diff * 20
        
        # Pawn structure penalties
        for c in range(8):
            # Doubled pawns
            if w_pawn_files[c] > 1:
                score -= (w_pawn_files[c] - 1) * 20
            if b_pawn_files[c] > 1:
                score += (b_pawn_files[c] - 1) * 20
            
            # Isolated pawns
            if w_pawn_files[c] > 0:
                has_neighbor = (c > 0 and w_pawn_files[c-1] > 0) or (c < 7 and w_pawn_files[c+1] > 0)
                if not has_neighbor:
                    score -= 15
            if b_pawn_files[c] > 0:
                has_neighbor = (c > 0 and b_pawn_files[c-1] > 0) or (c < 7 and b_pawn_files[c+1] > 0)
                if not has_neighbor:
                    score += 15
        
        # Draw detection
        if self.engine.is_draw():
            return 0
            
        return score if self.engine.turn == 'w' else -score

    def get_opening_move(self):
        # Join move history with spaces (e.g. "e2e4 e7e5 g1f3")
        history = ' '.join(self.engine.coordinate_history)
        
        # Use global OPENING_BOOK from config.py
        if history in OPENING_BOOK:
            choices = OPENING_BOOK[history]
            if choices:
                # Random choice for variety (rotation)
                move_str = random.choice(choices)
                move = self.engine.parse_move(move_str)
                if move:
                    legal = self.engine.get_all_legal_moves(self.engine.turn)
                    for lm in legal:
                        if lm['from'] == move['from'] and lm['to'] == move['to']:
                            print(f"  [AI] Book: {move_str} (Path: '{history}')")
                            return lm
        return None
