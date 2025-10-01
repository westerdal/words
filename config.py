#!/usr/bin/env python3
"""
Configuration management for the word game system
"""
import os
from typing import Optional

class Config:
    """Configuration class for managing environment variables and settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    
    # Game Configuration
    MAX_HINTS: int = 5
    DEFAULT_WORD_LIMIT: int = 10000
    
    # File Paths
    DATA_DIR: str = "data"
    SECRETWORD_DIR: str = "secretword"
    EMBEDDINGS_FILE: str = ".env/embeddings2.json"
    DICTIONARY_FILE: str = "data/enable2.txt"
    
    # Processing Configuration
    BATCH_SIZE: int = 50
    SAVE_INTERVAL: int = 200
    CONSECUTIVE_WEAK_THRESHOLD: int = 5
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment variables"""
        cls.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        cls.MAX_HINTS = int(os.getenv('MAX_HINTS', '5'))
        cls.DEFAULT_WORD_LIMIT = int(os.getenv('DEFAULT_WORD_LIMIT', '10000'))
        cls.DATA_DIR = os.getenv('DATA_DIR', 'data')
        cls.SECRETWORD_DIR = os.getenv('SECRETWORD_DIR', 'secretword')
        cls.EMBEDDINGS_FILE = os.getenv('EMBEDDINGS_FILE', '.env/embeddings2.json')
        cls.DICTIONARY_FILE = os.getenv('DICTIONARY_FILE', 'data/enable2.txt')
        cls.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
        cls.SAVE_INTERVAL = int(os.getenv('SAVE_INTERVAL', '200'))
        cls.CONSECUTIVE_WEAK_THRESHOLD = int(os.getenv('CONSECUTIVE_WEAK_THRESHOLD', '5'))
    
    @classmethod
    def load_from_file(cls, env_file: str = None):
        """Load configuration from a .env file"""
        # Try multiple file locations in order of preference
        if env_file is None:
            env_files = [".env", "config.env"]
        else:
            env_files = [env_file]
        
        for file_path in env_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                break  # Stop after loading the first found file
        
        # Load from environment after setting from file
        cls.load_from_env()
    
    @classmethod
    def validate_openai_key(cls) -> bool:
        """Validate that OpenAI API key is set"""
        return cls.OPENAI_API_KEY is not None and cls.OPENAI_API_KEY != "your_openai_api_key_here"
    
    @classmethod
    def get_openai_client(cls):
        """Get configured OpenAI client"""
        if not cls.validate_openai_key():
            raise ValueError("OpenAI API key not configured! Please set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            return OpenAI(api_key=cls.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

# Load configuration on import
Config.load_from_file()

# Convenience function
def get_openai_client():
    """Get configured OpenAI client"""
    return Config.get_openai_client()
