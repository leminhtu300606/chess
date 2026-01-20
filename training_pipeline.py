"""
Training Pipeline for Chess AI
Quy trình huấn luyện cho Chess AI
"""

import numpy as np
from supervised_learner import SupervisedLearner, SimpleNeuralNetwork, PositionEncoder
from pgn_importer import PGNImporter


class TrainingPipeline:
    """Manage the entire training process"""
    
    def __init__(self):
        self.learner = SupervisedLearner()
        self.encoder = PositionEncoder()
        self.neural_net = SimpleNeuralNetwork()
        self.importer = PGNImporter()
    
    def import_external_data(self, source_path, source_type='file'):
        """
        Import external training data
        
        Args:
            source_path: path to PGN file or directory
            source_type: 'file' or 'directory'
        
        Returns:
            number of games imported
        """
        print(f"\n[Pipeline] Importing external data: {source_path}")
        
        if source_type == 'file':
            count = self.importer.import_file(source_path)
        elif source_type == 'directory':
            count = self.importer.import_directory(source_path)
        else:
            print("[Pipeline] Unknown source type")
            count = 0
        
        return count
    
    def train_neural_network(self, epochs=100):
        """
        Train neural network on collected game data
        
        Args:
            epochs: number of training epochs
        """
        print(f"\n[Pipeline] Training neural network...")
        
        # Collect all learned moves
        moves = list(self.learner.training_data['move_stats'].keys())
        if not moves:
            print("[Pipeline] No moves data to train on")
            return False
        
        print(f"[Pipeline] Found {len(moves)} unique moves to learn from")
        
        # Create training targets (move quality scores)
        X_data = []
        y_data = []
        
        for move_key in moves[:1000]:  # Limit to top 1000 moves
            quality = self.learner.get_move_quality(move_key)
            y_data.append(quality)
            
            # For now, use random features (in full implementation, would use board state)
            X_data.append(np.random.randn(256))
        
        if len(X_data) > 0:
            print(f"[Pipeline] Training on {len(X_data)} examples...")
            self.neural_net.train(X_data, y_data, epochs=epochs, learning_rate=0.01)
            self.neural_net.save_model()
            print("[Pipeline] Neural network training complete!")
            return True
        
        return False
    
    def analyze_move_patterns(self):
        """Analyze learned move patterns and print insights"""
        print("\n[Pipeline] Analyzing move patterns...")
        
        moves = self.learner.training_data['move_stats']
        
        if not moves:
            print("[Pipeline] No move data to analyze")
            return
        
        # Find best and worst moves
        move_qualities = {}
        for move_key, stats in moves.items():
            total = stats['wins'] + stats['draws'] + stats['losses']
            if total >= 3:  # Need at least 3 occurrences
                quality = (stats['wins'] - stats['losses']) / total
                move_qualities[move_key] = quality
        
        if move_qualities:
            sorted_moves = sorted(move_qualities.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n[Pipeline] Top 10 Best Moves (win rate > 0):")
            for move_key, quality in sorted_moves[:10]:
                if quality > 0:
                    print(f"  {move_key}: {quality:.2%} win rate")
            
            print(f"\n[Pipeline] Top 10 Worst Moves (win rate < 0):")
            for move_key, quality in sorted(sorted_moves, key=lambda x: x[1])[:10]:
                if quality < 0:
                    print(f"  {move_key}: {quality:.2%} win rate")
            
            # Statistics
            qualities = list(move_qualities.values())
            print(f"\n[Pipeline] Overall Statistics:")
            print(f"  Average move quality: {np.mean(qualities):.2%}")
            print(f"  Median move quality: {np.median(qualities):.2%}")
            print(f"  Std deviation: {np.std(qualities):.2%}")
    
    def print_learning_stats(self):
        """Print learning progress statistics"""
        stats = self.learner.get_learning_stats()
        
        print("\n" + "="*50)
        print("LEARNING STATISTICS")
        print("="*50)
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
        print("="*50 + "\n")
    
    def save_checkpoint(self, name):
        """Save training checkpoint"""
        checkpoint = {
            'move_patterns': self.learner.training_data['move_stats'],
            'openings': self.learner.training_data['openings'],
            'endgames': self.learner.training_data['endgames']
        }
        
        filename = f"checkpoint_{name}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"[Pipeline] Saved checkpoint: {filename}")
        except Exception as e:
            print(f"[Pipeline] Error saving checkpoint: {e}")
    
    def run_full_training(self, pgn_source=None, epochs=100):
        """
        Run complete training pipeline
        
        Args:
            pgn_source: path to PGN file or directory (optional)
            epochs: number of neural network epochs
        """
        print("\n" + "="*50)
        print("CHESS AI TRAINING PIPELINE")
        print("="*50)
        
        # Step 1: Import external data if provided
        if pgn_source:
            import os
            if os.path.isdir(pgn_source):
                self.import_external_data(pgn_source, 'directory')
            else:
                self.import_external_data(pgn_source, 'file')
        
        # Step 2: Analyze patterns
        self.analyze_move_patterns()
        
        # Step 3: Train neural network
        if len(self.learner.training_data['move_stats']) > 50:
            self.train_neural_network(epochs=epochs)
        else:
            print("[Pipeline] Not enough move data for neural network training yet")
        
        # Step 4: Print final stats
        self.print_learning_stats()
        
        # Step 5: Save checkpoint
        self.save_checkpoint("full_training")
        
        print("[Pipeline] Training complete!")


def quick_import_demo():
    """Quick demo: import example PGN and show results"""
    print("\n[Demo] Chess AI Supervised Learning Pipeline")
    print("[Demo] =========================================\n")
    
    pipeline = TrainingPipeline()
    
    # Check if sample PGN exists
    sample_pgn = "sample_games.pgn"
    if os.path.exists(sample_pgn):
        print(f"[Demo] Found sample PGN file: {sample_pgn}")
        pipeline.run_full_training(pgn_source=sample_pgn, epochs=50)
    else:
        print(f"[Demo] No sample PGN found at {sample_pgn}")
        print("[Demo] To train on external data:")
        print("[Demo]   1. Get a PGN file (e.g., from Lichess or Chess.com)")
        print("[Demo]   2. Place it in the workspace folder")
        print("[Demo]   3. Run: pipeline.run_full_training('your_file.pgn')\n")
        
        pipeline.print_learning_stats()


if __name__ == "__main__":

    
    # Run demo
    quick_import_demo()
