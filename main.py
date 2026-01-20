import sys
import time
import pygame
from config import *
from engine import ChessEngine
from ai import ChessAI
from ai_rl import RLChessAI

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI Grandmaster")
        self.clock = pygame.time.Clock()
        self.engine = ChessEngine()
        self.ai = ChessAI(self.engine)
        self.ai_rl = RLChessAI(self.engine)
        
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
        self.font_bold = pygame.font.SysFont("Arial", 22, bold=True)
        
        # Menu State
        self.selected_mode = 'AI'
        self.ai_type = 'Supervised' # Supervised or RL
        self.selected_time = 600
        self.selected_side = 'w'
        
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
        game_ended = False
        result = None
        
        if self.engine.is_checkmate(color):
            self.status_message = f"Checkmate! {'White' if opp == 'w' else 'Black'} Wins!"
            result = 'win' if opp != self.player_side else 'loss'
            game_ended = True
        elif self.engine.is_stalemate(color):
            self.status_message = "Stalemate! Draw."
            result = 'draw'
            game_ended = True
        elif self.engine.is_draw():
            self.status_message = "Draw! (Insufficient Material or 50-Move Rule)"
            result = 'draw'
            game_ended = True

        # Trigger AI learning when game ends
        if game_ended and self.game_mode == 'AI':
            ai_color = 'b' if self.player_side == 'w' else 'w'
            self.ai.analyze_game(ai_color, result)
        
        return game_ended


    def start_game(self):
        self.game_state = "PLAYING"
        self.engine.reset()
        self.player_side = self.selected_side
        if self.selected_side == 'random':
            import random
            self.player_side = random.choice(['w', 'b'])
            
        self.game_mode = self.selected_mode
        self.time_limit = self.selected_time
        
        val = self.time_limit if self.time_limit != -1 else 999999
        self.timers = {'w': val, 'b': val}
        self.last_time = time.time() 
        self.status_message = ""

    def unified_menu(self):
        self.screen.fill(COLOR_BG)
        title = self.font_title.render("CHESS GRANDMASTER", True, COLOR_TEXT)
        self.screen.blit(title, (WINDOW_WIDTH//2 - 150, 50))
        
        mouse = pygame.mouse.get_pos()
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()

        def draw_option_group(title, options, selected_val, y_start, key_attr):
            self.screen.blit(self.font_bold.render(title, True, COLOR_HIGHLIGHT), (50, y_start))
            
            for i, (text, val) in enumerate(options):
                rect = pygame.Rect(50 + (i % 2) * 220, y_start + 40 + (i // 2) * 60, 200, 40)
                is_selected = (selected_val == val)
                item_col = (100, 200, 100) if is_selected else (60, 60, 60)
                
                if rect.collidepoint(mouse):
                    item_col = (80, 180, 80) if is_selected else (80, 80, 80)
                    for event in events: # Check events specifically for this button
                        if event.type == pygame.MOUSEBUTTONDOWN:
                             setattr(self, key_attr, val)

                pygame.draw.rect(self.screen, item_col, rect, border_radius=8)
                label = self.font_ui.render(text, True, COLOR_TEXT)
                self.screen.blit(label, (rect.x + 20, rect.y + 10))

        # 1. Game Mode
        draw_option_group("1. Game Mode", [("Player vs AI", 'AI'), ("Player vs Player", 'PvP')], 
                         self.selected_mode, 120, 'selected_mode')

        # 1b. AI Type (Only if AI mode selected)
        if self.selected_mode == 'AI':
            draw_option_group("1b. AI Type", [("Standard AI", 'Supervised'), ("Self-Play AI (RL)", 'RL')], 
                             self.selected_ai_type_dummy if hasattr(self, 'selected_ai_type_dummy') else self.ai_type, 200, 'ai_type')

        # 2. Time Control
        draw_option_group("2. Time Control", [("1 Minute", 60), ("3 Minutes", 180), ("10 Minutes", 600), ("Unlimited", -1)], 
                         self.selected_time, 330, 'selected_time')

        # 3. Side Selection (Only valid for AI)
        if self.selected_mode == 'AI':
            draw_option_group("3. Choose Side", [("White", 'w'), ("Black", 'b'), ("Random", 'random')], 
                             self.selected_side, 450, 'selected_side')
        
        # Start Button
        btn_start = pygame.Rect(WINDOW_WIDTH//2 - 100, 600, 200, 60)
        col_start = (255, 165, 0) # Orange
        if btn_start.collidepoint(mouse):
             col_start = (255, 140, 0)
             for event in events: # Check events specifically for this button
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.start_game()
        
        pygame.draw.rect(self.screen, col_start, btn_start, border_radius=15)
        start_txt = self.font_title.render("START GAME", True, (0,0,0))
        self.screen.blit(start_txt, (btn_start.x + 15, btn_start.y + 12))
        
        pygame.display.flip()

    def run(self):
        while True:
            if self.game_state == "MENU":
                self.unified_menu()
                continue
                
            dt = time.time() - self.last_time
            self.last_time = time.time()
            
            if self.engine.game_active and self.time_limit != -1:
                # Timer logic
                self.timers[self.engine.turn] -= dt
                if self.timers[self.engine.turn] <= 0:
                    self.engine.game_active = False
                    opp = 'b' if self.engine.turn == 'w' else 'w'
                    winner = "White" if opp == 'w' else "Black"
                    self.status_message = f"Time's Up! {winner} Wins!"

            self.screen.fill(COLOR_BG)
            self.draw_board()
            self.draw_ui()
            
            if self.selected_square:
                pygame.draw.rect(self.screen, COLOR_SELECTED, 
                               (self.selected_square[1]*CELL_SIZE, self.selected_square[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    
                    # Sidebar Buttons
                    if x > BOARD_SIZE:
                        # Undo
                        if BOARD_SIZE + 20 <= x <= BOARD_SIZE + 120 and 600 <= y <= 640:
                            if self.game_mode == 'AI':
                                # Undo twice for AI
                                self.engine.undo_move()
                                self.engine.undo_move()
                            else:
                                self.engine.undo_move()
                            self.selected_square = None
                            self.valid_moves = []
                        # Menu
                        elif BOARD_SIZE + 140 <= x <= BOARD_SIZE + 240 and 600 <= y <= 640:
                             self.game_state = "MENU"
                    
                    # Board Interaction
                    elif x < BOARD_SIZE:
                        c, r = x // CELL_SIZE, y // CELL_SIZE
                        
                        # Validate coordinates
                        if not (0 <= r < 8 and 0 <= c < 8):
                            continue
                        
                        if self.engine.game_active:
                            if (self.game_mode == 'PvP') or \
                               (self.game_mode == 'AI' and self.engine.turn == self.player_side):
                                
                                # Handle Moving
                                moved = False
                                for m in self.valid_moves:
                                    if m['to'] == (r, c):
                                        success = self.engine.make_move(m)
                                        if success:
                                            moved = True
                                            self.selected_square = None
                                            self.valid_moves = []
                                            if self.check_game_over(): self.engine.game_active = False
                                            
                                            # Redraw immediately to show move
                                            self.screen.fill(COLOR_BG)
                                            self.draw_board()
                                            self.draw_ui()
                                            pygame.display.flip()
                                            
                                            # Trigger AI
                                            if self.game_mode == 'AI' and self.engine.game_active:
                                                pygame.event.pump() # Prevent freeze
                                                time.sleep(0.1)
                                                current_ai = self.ai if self.ai_type == 'Supervised' else self.ai_rl
                                                aimove = current_ai.get_best_move()
                                                if aimove: 
                                                    ai_color = 'b' if self.player_side == 'w' else 'w'
                                                    # self.ai.record_position(aimove, ai_color)  # removed, not needed
                                                    self.engine.make_move(aimove)
                                                    if self.check_game_over(): self.engine.game_active = False
                                        break
                                
                                if not moved:
                                     # Select Piece
                                    p = self.engine.board[r][c]
                                    if p and p['color'] == self.engine.turn:
                                         self.selected_square = (r, c)
                                         self.valid_moves = self.engine.get_piece_moves(r, c)
                                    else:
                                         self.selected_square = None; self.valid_moves = []

            # AI Turn Trigger (for start of game if AI is White)
            if self.game_mode == 'AI' and self.engine.game_active and self.engine.turn != self.player_side:
                pygame.event.pump()
                current_ai = self.ai if self.ai_type == 'Supervised' else self.ai_rl
                aimove = current_ai.get_best_move()
                if aimove:
                    ai_color = 'b' if self.player_side == 'w' else 'w'
                    # self.ai.record_position(aimove, ai_color)  # removed, not needed
                    self.engine.make_move(aimove)
                    if self.check_game_over(): self.engine.game_active = False

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
