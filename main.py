import sys
import time
import pygame
from config import *
from engine import ChessEngine
from ai import ChessAI

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI Grandmaster")
        self.clock = pygame.time.Clock()
        self.engine = ChessEngine()
        self.ai = ChessAI(self.engine)
        
        self.selected_square = None
        self.valid_moves = []
        self.player_side = 'w' 
        self.game_mode = 'AI' 
        self.time_limit = 600 # Default 10m
        self.game_state = "MENU" # MENU -> TIME_SELECT -> PLAYING
        self.status_message = ""
        
        self.font_large = pygame.font.SysFont("Segoe UI Symbol", 60)
        self.font_ui = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_title = pygame.font.SysFont("Arial", 30, bold=True)
        
        self.timers = {'w': 600, 'b': 600}
        self.last_time = time.time()

    def draw_board(self):
        pygame.draw.rect(self.screen, (30, 30, 30), (BOARD_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        flipped = (self.player_side == 'b' and self.game_mode == 'AI') # Flip only in AI mode if playing black
        for r in range(8):
            for c in range(8):
                vr, vc = (7-r if flipped else r), (7-c if flipped else c)
                color = COLOR_BOARD_LIGHT if (r + c) % 2 == 0 else COLOR_BOARD_DARK
                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if (r, c) == self.selected_square:
                    color = COLOR_SELECTED
                
                # Draw Cell
                pygame.draw.rect(self.screen, color, rect)

                # Highlight last move (Yellowish overlay)
                if self.engine.last_move:
                    if (r, c) in [self.engine.last_move['from'], self.engine.last_move['to']]:
                         # Create a transparent surface for highlight
                         s = pygame.Surface((CELL_SIZE, CELL_SIZE))
                         s.set_alpha(100) # Transparency
                         s.fill(COLOR_HIGHLIGHT)
                         self.screen.blit(s, (c * CELL_SIZE, r * CELL_SIZE))
                
                # Hints
                for m in self.valid_moves:
                    if m['to'] == (r, c):
                        center = (c * CELL_SIZE + CELL_SIZE//2, r * CELL_SIZE + CELL_SIZE//2)
                        pygame.draw.circle(self.screen, (100, 100, 100), center, 10)

                # Piece
                piece = self.engine.board[r][c]
                if piece:
                    text_color = (255, 255, 255) if piece['color'] == 'w' else (0, 0, 0)
                    char = PIECE_UNICODE[piece['color'] + piece['type']]
                    surf = self.font_large.render(char, True, text_color)
                    rect_char = surf.get_rect(center=(c * CELL_SIZE + CELL_SIZE//2, r * CELL_SIZE + CELL_SIZE//2))
                    self.screen.blit(surf, rect_char)

    def draw_ui(self):
        # ... (Title existing) ...
        title = self.font_title.render("Chess AI", True, COLOR_TEXT)
        self.screen.blit(title, (BOARD_SIZE + 20, 20))
        
        # Timers
        t_w_str = "Unlimited" if self.time_limit == -1 else f"{int(self.timers['w'])//60}:{int(self.timers['w'])%60:02d}"
        t_b_str = "Unlimited" if self.time_limit == -1 else f"{int(self.timers['b'])//60}:{int(self.timers['b'])%60:02d}"
        
        t_w = f"White: {t_w_str}"
        t_b = f"Black: {t_b_str}"
        
        col_w = (255, 255, 255) if self.engine.turn == 'w' else (100, 100, 100)
        col_b = (255, 255, 255) if self.engine.turn == 'b' else (100, 100, 100)
        
        self.screen.blit(self.font_ui.render(t_w, True, col_w), (BOARD_SIZE + 20, 120))
        self.screen.blit(self.font_ui.render(t_b, True, col_b), (BOARD_SIZE + 20, 80))
        
        status = "Your Turn" 
        if self.game_mode == 'AI' and self.engine.turn != self.player_side: status = "AI Thinking..."
        if self.game_mode == 'PvP': status = "White's Turn" if self.engine.turn == 'w' else "Black's Turn"
        
        if not self.engine.game_active: status = self.status_message or "Game Over"
        col_status = (255, 100, 100) if "Checkmate" in status else COLOR_HIGHLIGHT
        self.screen.blit(self.font_ui.render(status, True, col_status), (BOARD_SIZE + 20, 160))

        # Move History
        pygame.draw.rect(self.screen, (40, 40, 40), (BOARD_SIZE + 10, 200, SIDEBAR_WIDTH - 20, 380))
        self.screen.blit(self.font_ui.render("Move History:", True, (200, 200, 200)), (BOARD_SIZE + 20, 210))
        
        history = self.engine.move_history[-14:] # Show last 14 ply (7 moves)
        for i, move_str in enumerate(history):
            # Format: "1. e4" or "1... e5" logic is hard without full context, just listing moves
            # We can use index to deduce.
            # Real full notation needs sync, here we just list strings.
            y_pos = 240 + i * 25
            turn_idx = len(self.engine.move_history) - len(history) + i
            # If 0 (first move) -> White.
            prefix = f"{(turn_idx//2 + 1)}." if turn_idx % 2 == 0 else "..."
            text = f"{prefix} {move_str}"
            self.screen.blit(self.font_small.render(text, True, COLOR_TEXT), (BOARD_SIZE + 30, y_pos))

        mouse = pygame.mouse.get_pos()
        # Undo
        btn_undo = pygame.Rect(BOARD_SIZE + 20, 600, 100, 40)
        col = COLOR_BTN_HOVER if btn_undo.collidepoint(mouse) else COLOR_BTN
        pygame.draw.rect(self.screen, col, btn_undo, border_radius=5)
        self.screen.blit(self.font_ui.render("Undo", True, COLOR_TEXT), (btn_undo.x + 25, btn_undo.y + 10))
        
        # Menu
        btn_menu = pygame.Rect(BOARD_SIZE + 140, 600, 100, 40)
        col2 = COLOR_BTN_HOVER if btn_menu.collidepoint(mouse) else COLOR_BTN
        pygame.draw.rect(self.screen, col2, btn_menu, border_radius=5)
        self.screen.blit(self.font_ui.render("Menu", True, COLOR_TEXT), (btn_menu.x + 25, btn_menu.y + 10))

    def check_game_over(self):
        color = self.engine.turn 
        opp = 'b' if color == 'w' else 'w'
        if self.engine.is_checkmate(color):
            self.status_message = f"Checkmate! {'White' if opp == 'w' else 'Black'} Wins!"
            return True
        if self.engine.is_stalemate(color):
            self.status_message = "Stalemate! Draw."
            return True
        return False

    def side_menu(self):
        self.screen.fill(COLOR_BG)
        title = self.font_title.render("SELECT SIDE", True, COLOR_TEXT)
        self.screen.blit(title, (WINDOW_WIDTH//2 - 100, 150))
        
        options = [
            ("Play as White", 'w', 250),
            ("Play as Black", 'b', 320),
            ("Random Side", 'random', 390)
        ]
        
        mouse = pygame.mouse.get_pos()
        
        for text, val, y in options:
            btn = pygame.Rect(WINDOW_WIDTH//2 - 120, y, 240, 50)
            col = COLOR_BTN_HOVER if btn.collidepoint(mouse) else COLOR_BTN
            pygame.draw.rect(self.screen, col, btn, border_radius=10)
            self.screen.blit(self.font_ui.render(text, True, COLOR_TEXT), (btn.x + 40, btn.y + 15))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for text, val, y in options:
                    btn = pygame.Rect(WINDOW_WIDTH//2 - 120, y, 240, 50)
                    if btn.collidepoint(event.pos):
                         if val == 'random':
                             import random
                             self.player_side = random.choice(['w', 'b'])
                         else:
                             self.player_side = val
                         self.game_state = "TIME_SELECT"

    def main_menu(self):
        self.screen.fill(COLOR_BG)
        title = self.font_title.render("CHESS GRANDMASTER", True, COLOR_TEXT)
        self.screen.blit(title, (WINDOW_WIDTH//2 - 150, 200))

        mouse = pygame.mouse.get_pos()
        
        # Define Buttons
        btn_ai = pygame.Rect(WINDOW_WIDTH//2 - 100, 300, 200, 50)
        btn_pvp = pygame.Rect(WINDOW_WIDTH//2 - 100, 370, 200, 50)

        # Draw Buttons
        col_ai = COLOR_BTN_HOVER if btn_ai.collidepoint(mouse) else COLOR_BTN
        pygame.draw.rect(self.screen, col_ai, btn_ai, border_radius=10)
        self.screen.blit(self.font_ui.render("Player vs AI", True, COLOR_TEXT), (btn_ai.x + 50, btn_ai.y + 15))

        col_pvp = COLOR_BTN_HOVER if btn_pvp.collidepoint(mouse) else COLOR_BTN
        pygame.draw.rect(self.screen, col_pvp, btn_pvp, border_radius=10)
        self.screen.blit(self.font_ui.render("Player vs Player", True, COLOR_TEXT), (btn_pvp.x + 40, btn_pvp.y + 15))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_ai.collidepoint(event.pos):
                    self.game_mode = 'AI'
                    self.game_state = "SIDE_SELECT" 
                elif btn_pvp.collidepoint(event.pos):
                    self.game_mode = 'PvP'
                    self.game_state = "TIME_SELECT"

    def start_game(self):
        self.game_state = "PLAYING"
        self.engine.reset()
        val = self.time_limit if self.time_limit != -1 else 999999
        self.timers = {'w': val, 'b': val}
        self.last_time = time.time() # Reset clock to prevent huge dt from menu wait
        self.status_message = ""

    def time_menu(self):
        self.screen.fill(COLOR_BG)
        title = self.font_title.render("SELECT TIME CONTROL", True, COLOR_TEXT)
        self.screen.blit(title, (WINDOW_WIDTH//2 - 150, 150))
        
        options = [
            ("1 Minute (Bullet)", 60, 250),
            ("3 Minutes (Blitz)", 180, 320),
            ("10 Minutes (Standard)", 600, 390),
            ("No Time Limit", -1, 460)
        ]
        
        mouse = pygame.mouse.get_pos()
        
        for text, val, y in options:
            btn = pygame.Rect(WINDOW_WIDTH//2 - 120, y, 240, 50)
            col = COLOR_BTN_HOVER if btn.collidepoint(mouse) else COLOR_BTN
            pygame.draw.rect(self.screen, col, btn, border_radius=10)
            self.screen.blit(self.font_ui.render(text, True, COLOR_TEXT), (btn.x + 20, btn.y + 15))
        
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                 for text, val, y in options:
                    btn = pygame.Rect(WINDOW_WIDTH//2 - 120, y, 240, 50)
                    if btn.collidepoint(event.pos):
                        self.time_limit = val
                        self.start_game()
        
        # If Player is Black and AI mode, AI moves first? 
        # Actually AI triggers in loop. 
        # But we need to make sure loop triggers AI if turn='w' and player='b'
        # Check run loop logic below.

    # ... time_menu ...

    def run(self):
        while True:
            if self.game_state == "MENU":
                self.main_menu()
                continue
            if self.game_state == "SIDE_SELECT":
                self.side_menu()
                continue
            if self.game_state == "TIME_SELECT":
                self.time_menu()
                continue
                
            dt = time.time() - self.last_time
            self.last_time = time.time()
            
            if self.engine.game_active and self.time_limit != -1:
                if self.engine.turn == 'w': self.timers['w'] -= dt
                else: self.timers['b'] -= dt
                if self.timers['w'] <= 0 or self.timers['b'] <= 0:
                    self.engine.game_active = False
                    self.status_message = "Time Out!"

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # UI Buttons
                    if x > BOARD_SIZE:
                        if 600 <= y <= 640:
                            if BOARD_SIZE+20 <= x <= BOARD_SIZE+120:
                                self.engine.undo_move(); self.engine.undo_move()
                            elif BOARD_SIZE+140 <= x <= BOARD_SIZE+240:
                                self.game_state = "MENU"
                    # Board Click
                    elif x < BOARD_SIZE:
                        c, r = x // CELL_SIZE, y // CELL_SIZE
                        
                        # In AI mode, if player is Black, board might be flipped. 
                        # We handle visual flip in draw_board.
                        # But clicks need to be mapped if flipped.
                        # draw_board logic: flipped = (self.player_side == 'b' and self.game_mode == 'AI')
                        flipped = (self.player_side == 'b' and self.game_mode == 'AI')
                        if flipped:
                            r, c = 7-r, 7-c

                        # Check Turn Validity
                        is_my_turn = True
                        if self.game_mode == 'AI' and self.engine.turn != self.player_side:
                            is_my_turn = False
                        
                        if self.engine.game_active and is_my_turn:
                             # ... (Existing click logic) ...
                             pass 
                             # Note: Since I am replacing the block, I need to include the click logic or rely on partial.
                             # But let's be careful. The replacement chunk handles the loop structure. 
                             # I should include the click handling inside.
                             if self.selected_square:
                                # Try Move
                                move = next((m for m in self.valid_moves if m['to'] == (r, c)), None)
                                if move:
                                    self.engine.make_move(move)
                                    self.selected_square = None
                                    self.valid_moves = []
                                    self.draw_board(); pygame.display.flip() # Draw BEFORE AI thinks
                                    
                                    if self.check_game_over(): 
                                        self.engine.game_active = False
                                    elif self.game_mode == 'AI':
                                        # TRIGGER AI
                                        # Force refresh before sleep
                                        self.draw_board()
                                        self.draw_ui()
                                        pygame.display.flip()
                                        
                                        time.sleep(0.1)
                                        aimove = self.ai.get_best_move()
                                        if aimove: 
                                            self.engine.make_move(aimove)
                                            if self.check_game_over(): self.engine.game_active = False
                                else:
                                    p = self.engine.board[r][c]
                                    if p and p['color'] == self.engine.turn:
                                        self.selected_square = (r, c)
                                        self.valid_moves = self.engine.get_piece_moves(r, c)
                                    else:
                                        self.selected_square = None; self.valid_moves = []
                             else:
                                p = self.engine.board[r][c]
                                if p and p['color'] == self.engine.turn:
                                     self.selected_square = (r, c)
                                     self.valid_moves = self.engine.get_piece_moves(r, c)

            # AI Turn Trigger (for when AI is White/First or after Player moves)
            # Actually, the Click event triggers AI response.
            # But if Player is Black, AI is White. AI needs to move FIRST.
            # We need a check outside event loop.
            if self.game_mode == 'AI' and self.engine.game_active and self.engine.turn != self.player_side:
                 # AI Turn (e.g. Start of game if AI=White)
                 # We need to make sure we don't spam.
                 # Only trigger if we didn't just move (handled by click)? 
                 # No, click handles Player move. This handles AI move if it's AI turn and NO events happened.
                 # Wait, if we use sleep here, we block events.
                 # Better: Draw frame, then move.
                 self.draw_board()
                 self.draw_ui()
                 pygame.display.flip()
                 time.sleep(0.1) # Small delay for UX
                 aimove = self.ai.get_best_move()
                 if aimove: 
                     self.engine.make_move(aimove)
                     if self.check_game_over(): self.engine.game_active = False

            self.screen.fill(COLOR_BG)
            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
