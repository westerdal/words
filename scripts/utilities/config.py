#!/usr/bin/env python3
"""
Centralized configuration for Semantic Rank Word Processing System
"""

import os
from pathlib import Path

class Config:
    """Configuration settings for the entire system"""
    
    # === PATHS ===
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    SECRETWORD_DIR = ROOT_DIR / "secretword" 
    LOGS_DIR = ROOT_DIR / "logs"
    ENV_DIR = ROOT_DIR / ".env"
    TEMP_DIR = ROOT_DIR / "temp"
    
    # Word lists
    ENABLE1_FILE = DATA_DIR / "enable1.txt"
    ENABLE2_FILE = DATA_DIR / "enable2.txt"
    
    # Embeddings
    EMBEDDINGS_FILE = ENV_DIR / "embeddings.json"
    EMBEDDINGS2_FILE = ENV_DIR / "embeddings2.json"
    
    # === PROCESSING SETTINGS ===
    AI_BATCH_SIZE = 50
    CHECKPOINT_INTERVAL = 200  # Save every N words
    STATUS_UPDATE_INTERVAL = 30  # Seconds between status updates
    
    # AI Cutoff Settings
    WEAK_CONNECTION_THRESHOLD = 10  # Stop after N consecutive weak connections
    HARD_CUTOFF_RANK = 5000        # Stop AI calls after this rank (absolute maximum)
    MIN_AI_CLUES = 2500           # Minimum clues before allowing dynamic cutoff (2500, not 25000)
    
    # === AI SETTINGS ===
    OPENAI_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-large"
    MAX_CLUE_WORDS = 7
    
    # Recursive Expansion Settings
    EXPANSION_MAX_LEVELS = 2  # How deep to recurse (back to 2 levels)
    EXPANSION_MAX_WORDS_PER_LEVEL = 999999  # No limit on words per level
    EXPANSION_MAX_PER_WORD = 8  # Max expansions per source word
    EXPANSION_MIN_SIMILARITY = 0.4  # Only expand high-similarity words
    EXPANSION_BATCH_SIZE = 5  # Words per OpenAI call
    EXPANSION_GENERIC_FILTER = ['thing', 'item', 'stuff', 'object', 'things', 'items']  # Skip these
    
    # === FILE FORMATS ===
    CSV_COLUMNS = ["rank", "secret_word", "word", "clue", "connection_strength"]
    
    # CSV filename format: secretword-[difficulty]-[category]-[word].csv
    @staticmethod
    def get_csv_filename(difficulty, category, word):
        return f"secretword-{difficulty}-{category}-{word}.csv"
    
    # Embeddings filename format: embeddings-[word].txt
    @staticmethod
    def get_embeddings_filename(word):
        return f"embeddings-{word}.txt"
    
    # Progress filename format: [word]_progress.json
    @staticmethod
    def get_progress_filename(word):
        return Config.LOGS_DIR / f"{word}_progress.json"
    
    # Queue filename format: [word]_weak_queue.json
    @staticmethod
    def get_queue_filename(word):
        return Config.SECRETWORD_DIR / f"secretword-easy-animals-{word}_weak_queue.json"
    
    # === ENVIRONMENT CHECKS ===
    @staticmethod
    def check_openai_key():
        """Check if OpenAI API key is available"""
        return bool(os.getenv('OPENAI_API_KEY'))
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist"""
        for directory in [Config.LOGS_DIR, Config.SECRETWORD_DIR]:
            directory.mkdir(exist_ok=True)
    
    # === VALIDATION ===
    @staticmethod
    def validate_word(word):
        """Validate that a word is suitable for processing"""
        if not word or not word.strip():
            return False, "Word cannot be empty"
        
        word = word.strip().lower()
        
        if not word.isalpha():
            return False, "Word must contain only letters"
        
        if len(word) < 2:
            return False, "Word must be at least 2 characters"
        
        return True, word
    
    @staticmethod
    def get_file_paths(word):
        """Get all relevant file paths for a word"""
        word = word.lower().strip()
        
        return {
            'csv': Config.SECRETWORD_DIR / Config.get_csv_filename("easy", "animals", word),
            'embeddings': Config.SECRETWORD_DIR / Config.get_embeddings_filename(word),
            'progress': Config.get_progress_filename(word),
            'queue': Config.get_queue_filename(word)
        }

# === CONSTANTS ===
# Connection strength categories
CONNECTION_STRENGTHS = {
    'secret_word': 'secret_word',
    'strong': 'strong', 
    'medium': 'medium',
    'weak': 'weak',
    'hard_cutoff': 'hard_cutoff'
}

# Special clues
SPECIAL_CLUES = {
    'secret_word': 'This is the *.',
    'error': 'ERROR',
    'weak_connection': 'weak connection',
    'null': None
}

if __name__ == "__main__":
    # Test configuration
    print("=== Semantic Rank Configuration ===")
    print(f"Root Directory: {Config.ROOT_DIR}")
    print(f"Data Directory: {Config.DATA_DIR}")
    print(f"Secretword Directory: {Config.SECRETWORD_DIR}")
    print(f"Logs Directory: {Config.LOGS_DIR}")
    print(f"OpenAI Key Available: {Config.check_openai_key()}")
    
    # Test word validation
    test_words = ["forest", "123", "", "a", "test-word"]
    print("\n=== Word Validation Tests ===")
    for word in test_words:
        valid, result = Config.validate_word(word)
        print(f"'{word}' -> Valid: {valid}, Result: '{result}'")
    
    # Test file paths
    print("\n=== File Paths for 'forest' ===")
    paths = Config.get_file_paths("forest")
    for key, path in paths.items():
        print(f"{key}: {path}")
    
    print("\n✅ Configuration loaded successfully!")
