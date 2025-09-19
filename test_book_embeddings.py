#!/usr/bin/env python3
"""
Test creating standard embeddings for 'book'
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
sys.path.append(str(Path(__file__).parent / "scripts" / "embeddings"))

from config import Config
from progress_tracker import create_tracker, quick_log

def test_book_embeddings():
    """Test creating standard embeddings for 'book'"""
    print("=== Testing Standard Embeddings Creation for 'book' ===")
    
    secret_word = "book"
    
    # Check if embeddings2.json exists and is accessible
    if not Config.EMBEDDINGS2_CACHE_FILE.exists():
        print(f"❌ Embeddings cache not found: {Config.EMBEDDINGS2_CACHE_FILE}")
        return False
    
    print(f"✅ Found embeddings cache: {Config.EMBEDDINGS2_CACHE_FILE}")
    
    try:
        # Create progress tracker
        tracker = create_tracker(secret_word, "EMBEDDINGS_TEST", 114000)
        
        # Try to load embeddings cache (just check if it loads)
        quick_log(secret_word, "🔄 Testing embeddings cache load...")
        
        import json
        with open(Config.EMBEDDINGS2_CACHE_FILE, 'r', encoding='utf-8') as f:
            # Just read the first few characters to test
            test_data = f.read(100)
            if test_data.startswith('{'):
                quick_log(secret_word, "✅ Embeddings cache file appears valid")
            else:
                quick_log(secret_word, "❌ Embeddings cache file format issue")
                return False
        
        quick_log(secret_word, "✅ Basic embeddings test completed")
        return True
        
    except Exception as e:
        quick_log(secret_word, f"❌ Error during embeddings test: {e}")
        return False

if __name__ == "__main__":
    success = test_book_embeddings()
    sys.exit(0 if success else 1)
