
from engine import ChessEngine

def test_engine():
    e = ChessEngine()
    print("Reset Board")
    e.reset()
    
    # White e4
    print("Move e2e4")
    move = e.parse_move('e2e4')
    if not e.make_move(move):
        print("ERROR: Make e2e4 failed")
        return
        
    e.print_board_debug() # Need to add this or inspect
    
    # Check Black Moves
    print("Listing Black Moves:")
    moves = e.get_all_legal_moves('b')
    print(f"Total Black Moves: {len(moves)}")
    if not moves:
        print("FATAL: No moves for Black after e2e4!")
        print(f"Is Check? {e.is_in_check('b')}")
    else:
        print("Moves found. Sample:", [f"{m['from']}->{m['to']}" for m in moves[:5]])
        
    # Check Undo
    e.undo_move()
    print("Undid e4. Board state check...")
    # Verify e4 is empty and e2 has pawn... 
    
    print("Test Complete")

if __name__ == "__main__":
    test_engine()
