
import sys
import os

# Add the project directory to sys.path
sys.path.append(r"h:\tu")

from engine import ChessEngine
from ai import ChessAI, SIMPLE_VALUES

def test_integrated_penalties():
    engine = ChessEngine()
    ai = ChessAI(engine)
    
    # --- 1. Test Integrated Blocking Penalty ---
    # Setup: Rook on a1, White Knight on b1.
    engine.board = [[None for _ in range(8)] for _ in range(8)]
    engine.board[7][0] = {'type': 'r', 'color': 'w'} # Rook a1
    engine.board[7][1] = {'type': 'n', 'color': 'w'} # Knight b1
    engine.board[7][4] = {'type': 'k', 'color': 'w'} # White King
    engine.board[0][4] = {'type': 'k', 'color': 'b'} # Black King
    engine.turn = 'w'
    
    # Force turn to 'w' so evaluate() returns white's score
    eval_blocking_w = ai.evaluate()
    
    # Move Knight to c3 (not blocking)
    engine.board[7][1] = None
    engine.board[5][2] = {'type': 'n', 'color': 'w'}
    eval_not_blocking_w = ai.evaluate()
    
    print(f"DEBUG: White Knight at b1 (blocking) eval: {eval_blocking_w}")
    print(f"DEBUG: White Knight at c3 (not blocking) eval: {eval_not_blocking_w}")
    
    # Knight value 300, 2/3 is 200.
    if eval_not_blocking_w > eval_blocking_w:
        print(f"SUCCESS: Integrated blocking penalty detected. Diff: {eval_not_blocking_w - eval_blocking_w}")
    else:
        print(f"FAILURE: Integrated blocking penalty not detected. Diff: {eval_not_blocking_w - eval_blocking_w}")

    # --- 2. Test Castling Penalty (Decision logic) ---
    engine.reset()
    engine.board[7][1] = None # b1
    engine.board[7][2] = None # c1
    engine.board[7][3] = None # d1
    engine.board[7][5] = None # f1
    engine.board[7][6] = None # g1
    engine.turn = 'w'
    
    move_no_castle = {'from': (6, 0), 'to': (5, 0)} # a2 -> a3
    score_no_castle = ai._evaluate_tactical_move(move_no_castle)
    
    engine.castling_rights['w']['k'] = False
    engine.castling_rights['w']['q'] = False
    score_no_castle_no_rights = ai._evaluate_tactical_move(move_no_castle)
    
    print(f"DEBUG: Choice a3 with rights: {score_no_castle}")
    print(f"DEBUG: Choice a3 without rights: {score_no_castle_no_rights}")
    
    if score_no_castle < score_no_castle_no_rights - 6000:
        print("SUCCESS: Missed castling penalty applied correctly.")
    else:
        print("FAILURE: Missed castling penalty failed.")

if __name__ == "__main__":
    test_integrated_penalties()
