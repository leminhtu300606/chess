from engine import ChessEngine
try:
    e = ChessEngine()
    m = e.parse_move('e2e4')
    print(f"Result: {m}")
except Exception as err:
    print(f"Error: {err}")
