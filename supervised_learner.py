"""
Supervised Learning Module for Chess AI
Mô-đun học có giám sát cho Chess AI
"""

import json
import os
import numpy as np
from collections import defaultdict
import pickle
from config import PIECE_VALUES

# Define simple values if not in config
SIMPLE_VALUES = {'p': 100, 'n': 300, 'b': 300, 'r': 500, 'q': 900, 'k': 10000}


class PositionEncoder:
    """Encode board positions into feature vectors for neural network"""
    
    def __init__(self):
        self.feature_size = 256
    
    def encode_position(self, board, color='w'):
        """
        Encode chess position into a numeric feature vector
        
        Args:
            board: 8x8 board state
            color: perspective color ('w' or 'b')
        
        Returns:
            numpy array of shape (256,) with normalized features
        """
        features = []
        
        # 1. Piece placement (64 features - one per square)
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece is None:
                    features.append(0)
                else:
                    # Encode as: piece_type (1-6) * sign (+/-)
                    piece_values = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
                    val = piece_values.get(piece['type'], 0)
                    val = val if piece['color'] == color else -val
                    features.append(val / 6.0)  # Normalize to [-1, 1]
        
        # 2. Material balance (1 feature)
        white_material = sum(
            SIMPLE_VALUES.get(board[r][c]['type'], 0) 
            for r in range(8) for c in range(8)
            if board[r][c] and board[r][c]['color'] == 'w'
        )
        black_material = sum(
            SIMPLE_VALUES.get(board[r][c]['type'], 0) 
            for r in range(8) for c in range(8)
            if board[r][c] and board[r][c]['color'] == 'b'
        )
        material_balance = (white_material - black_material) / 3900.0
        if color == 'b':
            material_balance = -material_balance
        features.append(material_balance)
        
        # 3. Pawn structure (8 features - pawns per file)
        for file_idx in range(8):
            pawn_count = 0
            for r in range(8):
                piece = board[r][file_idx]
                if piece and piece['type'] == 'p':
                    pawn_count += 1 if piece['color'] == color else -1
            features.append(pawn_count / 8.0)
        
        # 4. Piece counts (6 features)
        for piece_type in ['n', 'b', 'r', 'q', 'k']:
            count = 0
            for r in range(8):
                for c in range(8):
                    piece = board[r][c]
                    if piece and piece['type'] == piece_type:
                        count += 1 if piece['color'] == color else -1
            features.append(count / 16.0)
        
        # Pad to exactly 256 features
        while len(features) < 256:
            features.append(0)
        
        return np.array(features[:256], dtype=np.float32)


class SupervisedLearner:
    """
    Main supervised learning system
    Tracks move quality, position evaluations, and patterns
    """
    
    def __init__(self, data_file="chess_ai_training_data.json"):
        self.data_file = data_file
        self.encoder = PositionEncoder()
        
        # Training data storage
        self.training_data = {
            'positions': [],           # Encoded positions
            'position_evals': {},      # FEN -> evaluation
            'move_stats': {},          # move_key -> {wins, losses, draws}
            'openings': {},            # opening_name -> moves list
            'endgames': {},            # endgame_type -> positions
            'pgn_data_imported': False
        }
        
        self.load_training_data()
    
    def load_training_data(self):
        """Load training data from disk"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    self.training_data = json.load(f)
                    print(f"[Learning] Loaded {len(self.training_data['positions'])} training positions")
            except Exception as e:
                print(f"[Learning] Error loading training data: {e}")
    
    def save_training_data(self):
        """Save training data to disk"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"[Learning] Error saving training data: {e}")
    
    def record_move(self, move_key, result):
        """
        Record move outcome
        
        Args:
            move_key: "from_square->to_square"
            result: 1 (win), 0 (draw), -1 (loss)
        """
        if move_key not in self.training_data['move_stats']:
            self.training_data['move_stats'][move_key] = {'wins': 0, 'draws': 0, 'losses': 0}
        
        stats = self.training_data['move_stats'][move_key]
        if result == 1:
            stats['wins'] += 1
        elif result == 0:
            stats['draws'] += 1
        else:
            stats['losses'] += 1
    
    def get_move_quality(self, move_key):
        """
        Get quality score of a move based on history
        
        Returns:
            float in range [-1, 1] where 1 = excellent, -1 = terrible
        """
        if move_key not in self.training_data['move_stats']:
            return 0.0
        
        stats = self.training_data['move_stats'][move_key]
        total = stats['wins'] + stats['draws'] + stats['losses']
        
        if total == 0:
            return 0.0
        
        # Win rate calculation
        win_rate = (stats['wins'] - stats['losses']) / total
        return win_rate
    
    def import_pgn_games(self, pgn_text):
        """
        Import games from PGN format
        
        Args:
            pgn_text: string containing one or more games in PGN format
        """
        try:
            import chess
            import chess.pgn
            from io import StringIO
            
            pgn = StringIO(pgn_text)
            game_count = 0
            position_samples = []
            result_labels = []
            
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                game_count += 1
                
                # Extract result
                result = game.headers.get("Result", "1/2-1/2")
                if result == "1-0":
                    label = 1
                elif result == "0-1":
                    label = -1
                else:
                    label = 0
                
                # Get move sequence
                board = chess.Board()
                for move in game.mainline_moves():
                    move_san = board.san(move)
                    move_uci = move.uci()
                    
                    # Record this move
                    move_key = f"{move_uci[:2]}->{move_uci[2:]}"
                    self.record_move(move_key, label)
                    
                    # Encode board position and save sample
                    board_matrix = self._board_to_matrix(board)
                    encoded = self.encoder.encode_position(board_matrix, color='w' if board.turn else 'b')
                    position_samples.append(encoded.tolist())
                    result_labels.append(label)
                    # ...existing code...
                    board.push(move)
            
            # Lưu dữ liệu trạng thái bàn cờ vào training_data['positions']
            self.training_data['positions'].extend([
                {'features': feat, 'label': lbl} for feat, lbl in zip(position_samples, result_labels)
            ])
            self.training_data['pgn_data_imported'] = True
            print(f"[Learning] Imported {game_count} games from PGN")
            return game_count
        
        except ImportError:
            print("[Learning] python-chess library not found. Install with: pip install python-chess")
            return 0
        except Exception as e:
            print(f"[Learning] Error importing PGN: {e}")
            return 0
        def _board_to_matrix(self, board):
            """
            Chuyển đổi chess.Board thành ma trận 8x8 với dict {'type', 'color'} hoặc None
            """
            matrix = [[None for _ in range(8)] for _ in range(8)]
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    row = 7 - chess.square_rank(square)
                    col = chess.square_file(square)
                    matrix[row][col] = {'type': piece.symbol().lower(), 'color': 'w' if piece.color else 'b'}
            return matrix
        def train_deep_model(self, epochs=100, batch_size=64, learning_rate=0.01):
            """
            Huấn luyện mô hình học sâu trên dữ liệu trạng thái bàn cờ
            """
            if not self.training_data['positions']:
                print("[DeepLearning] Không có dữ liệu trạng thái bàn cờ để huấn luyện.")
                return
            X = [sample['features'] for sample in self.training_data['positions']]
            y = [sample['label'] for sample in self.training_data['positions']]
            nn = SimpleNeuralNetwork(input_size=256)
            nn.train(X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
            nn.save_model()
            print("[DeepLearning] Đã huấn luyện xong mô hình học sâu từ trạng thái bàn cờ.")
    
    def import_pgn_file(self, filepath):
        """
        Import games from a PGN file
        
        Args:
            filepath: path to .pgn file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                pgn_text = f.read()
            return self.import_pgn_games(pgn_text)
        except Exception as e:
            print(f"[Learning] Error reading PGN file: {e}")
            return 0
    
    def tag_opening(self, name, moves):
        """
        Tag an opening variation
        
        Args:
            name: opening name (e.g., "Sicilian Defense")
            moves: list of moves in format [from_square, to_square]
        """
        self.training_data['openings'][name] = {
            'moves': moves,
            'count': self.training_data['openings'].get(name, {}).get('count', 0) + 1
        }
    
    def tag_endgame(self, endgame_type, position_eval, objective):
        """
        Tag an endgame position with correct technique
        
        Args:
            endgame_type: e.g., "King and Pawn vs King"
            position_eval: position evaluation/score
            objective: e.g., "Promote pawn" or "Checkmate"
        """
        if endgame_type not in self.training_data['endgames']:
            self.training_data['endgames'][endgame_type] = []
        
        self.training_data['endgames'][endgame_type].append({
            'eval': position_eval,
            'objective': objective
        })
    
    def get_learning_stats(self):
        """Get statistics about learned patterns"""
        stats = {
            'total_moves_learned': len(self.training_data['move_stats']),
            'openings_tagged': len(self.training_data['openings']),
            'endgames_tagged': len(self.training_data['endgames']),
            'pgn_data_imported': self.training_data['pgn_data_imported']
        }
        
        # Calculate average win rate
        win_rates = []
        for move_key, move_stats in self.training_data['move_stats'].items():
            total = move_stats['wins'] + move_stats['draws'] + move_stats['losses']
            if total > 5:  # Only consider moves with enough data
                wr = (move_stats['wins'] - move_stats['losses']) / total
                win_rates.append(wr)
        
        if win_rates:
            stats['average_move_quality'] = np.mean(win_rates)
        
        return stats


class SimpleNeuralNetwork:
    """Simple neural network for position evaluation (optional)"""
    
    def __init__(self, input_size=256):
        self.input_size = input_size
        self.hidden_size = 128
        
        # Initialize with small random weights
        self.w1 = np.random.randn(input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        
        self.model_file = "chess_nn_model.pkl"
        self.load_model()
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """Forward pass"""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.output = self.sigmoid(self.z2)
        return self.output
    
    def backward(self, x, y, learning_rate=0.01):
        """Backpropagation"""
        m = x.shape[0]
        
        dz2 = self.output - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32):
        """Train network on data"""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        y = (y + 1) / 2  # Normalize to [0, 1]
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            if (epoch + 1) % 20 == 0:
                loss = np.mean((self.output - y) ** 2)
                print(f"[NN] Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, x):
        """Predict position value"""
        x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        output = self.forward(x)
        return (output * 2 - 1).flatten()
    
    def save_model(self):
        """Save model weights"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'w1': self.w1, 'b1': self.b1,
                    'w2': self.w2, 'b2': self.b2
                }, f)
        except Exception as e:
            print(f"[NN] Error saving model: {e}")
    
    def load_model(self):
        """Load model weights"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.w1 = data['w1']
                    self.b1 = data['b1']
                    self.w2 = data['w2']
                    self.b2 = data['b2']
            except Exception as e:
                print(f"[NN] Error loading model: {e}")
