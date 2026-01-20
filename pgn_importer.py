"""
PGN Importer for Chess AI
Tải dữ liệu từ file PGN (Portable Game Notation)
"""

import os
from supervised_learner import SupervisedLearner


class PGNImporter:
    """Import games from PGN files to train the AI"""
    
    def __init__(self):
        self.learner = SupervisedLearner()
    
    def import_file(self, filepath, limit=None):
        """
        Import all games from a PGN file
        
        Args:
            filepath: path to .pgn file
            limit: maximum number of games to import (None = all)
        
        Returns:
            number of games imported
        """
        if not os.path.exists(filepath):
            print(f"[PGN] File not found: {filepath}")
            return 0
        
        print(f"[PGN] Importing from: {filepath}")
        count = self.learner.import_pgn_file(filepath)
        
        if count > 0:
            self.learner.save_training_data()
            print(f"[PGN] Successfully imported {count} games")
        
        return count
    
        def download_pgn_examples():
            """Download example PGN files from online sources"""
            pass  # Placeholder for future implementation
        
        for pgn_file in pgn_files:
            print(f"\n[PGN] Processing: {os.path.basename(pgn_file)}")
            count = self.learner.import_pgn_file(pgn_file)
            total_games += count
        
        if total_games > 0:
            self.learner.save_training_data()
            print(f"\n[PGN] Total games imported: {total_games}")
        
        return total_games
    
    def get_stats(self):
        """Get import statistics"""
        return self.learner.get_learning_stats()


def download_pgn_examples():
    """
    Download example PGN files from online sources
    Requires: requests library
    """
    try:
        import requests
        
        sources = {
            'lichess': 'https://lichess.org/api/games/user/thibault?perfType=blitz&max=500',
        }
        
        print("[PGN] Attempting to download PGN files...")
        # Note: This is an example. Actual implementation would depend on API
        print("[PGN] For Lichess: export games directly from your profile")
        print("[PGN] For Chess.com: use https://www.chess.com/games/download")
        
    except ImportError:
        print("[PGN] requests library not found")


if __name__ == "__main__":
    importer = PGNImporter()
    split_dir = "h:/tu/pgn_split"
    if os.path.exists(split_dir):
        # Import tất cả file .pgn trong thư mục pgn_split
        for fname in os.listdir(split_dir):
            if fname.endswith('.pgn'):
                fpath = os.path.join(split_dir, fname)
                importer.import_file(fpath)
        # Xóa file pgn_example.pgn nếu tồn tại
        pgn_example = "h:/tu/pgn_example.pgn"
        if os.path.exists(pgn_example):
            os.remove(pgn_example)
            print(f"[PGN] Đã xóa file {pgn_example}")
    else:
        print(f"[PGN] Không tìm thấy thư mục {split_dir}")
    # Get statistics
    stats = importer.get_stats()
    print("\nLearning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
