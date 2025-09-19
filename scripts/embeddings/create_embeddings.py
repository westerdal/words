#!/usr/bin/env python3
"""
Generate ordered embeddings file for a secret word
Creates: secretword/embeddings-[word].txt with all words ranked by semantic similarity
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add utilities to path
sys.path.append(str(Path(__file__).parent.parent / "utilities"))
from config import Config
from progress_tracker import create_tracker, quick_log

class EmbeddingsGenerator:
    """Generates ordered embeddings for a secret word"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        
        # Validate word
        valid, result = Config.validate_word(self.secret_word)
        if not valid:
            raise ValueError(f"Invalid secret word: {result}")
        
        self.secret_word = result
        
        # File paths
        self.paths = Config.get_file_paths(self.secret_word)
        
        # Progress tracker
        self.tracker = None
        
        # Data
        self.embeddings_cache = {}
        self.enable2_words = []
        self.ranked_words = []
    
    def load_word_list(self) -> bool:
        """Load ENABLE2 word list"""
        quick_log(self.secret_word, f"📂 Loading word list from {Config.ENABLE2_FILE}")
        
        try:
            with open(Config.ENABLE2_FILE, 'r', encoding='utf-8') as f:
                self.enable2_words = [word.strip().lower() for word in f.readlines() if word.strip()]
            
            quick_log(self.secret_word, f"✅ Loaded {len(self.enable2_words):,} words from ENABLE2")
            
            if self.secret_word not in self.enable2_words:
                quick_log(self.secret_word, f"⚠️ WARNING: '{self.secret_word}' not found in ENABLE2 word list")
                return False
            
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to load word list: {e}")
            return False
    
    def load_embeddings_cache(self) -> bool:
        """Load embeddings from cache with progress updates"""
        # Check file size and provide time estimate
        try:
            file_size_mb = Config.EMBEDDINGS2_FILE.stat().st_size / (1024 * 1024)
            
            if file_size_mb < 100:
                time_estimate = "10-30 seconds"
            elif file_size_mb < 1000:
                time_estimate = "1-2 minutes"  
            elif file_size_mb < 3000:
                time_estimate = "3-5 minutes"
            else:
                time_estimate = "5-10 minutes"
            
            quick_log(self.secret_word, f"📂 Loading embeddings from {Config.EMBEDDINGS2_FILE}")
            quick_log(self.secret_word, f"📊 File size: {file_size_mb:.1f} MB")
            quick_log(self.secret_word, f"⏰ ESTIMATED TIME: {time_estimate}")
            quick_log(self.secret_word, f"🔄 Status updates will appear every 30 seconds...")
            quick_log(self.secret_word, f"⚠️  Please be patient - this is a large file!")
            
        except Exception as e:
            quick_log(self.secret_word, f"📂 Loading embeddings from {Config.EMBEDDINGS2_FILE}")
            quick_log(self.secret_word, f"⏳ This may take several minutes for large cache files...")
        
        try:
            import threading
            import time
            
            # Status update thread
            loading_complete = threading.Event()
            
            def status_updater():
                seconds = 0
                while not loading_complete.wait(30):  # Wait 30 seconds or until complete
                    seconds += 30
                    quick_log(self.secret_word, f"🔄 Still loading embeddings cache... ({seconds}s elapsed)")
            
            # Start status thread
            status_thread = threading.Thread(target=status_updater, daemon=True)
            status_thread.start()
            
            # Load the JSON file
            with open(Config.EMBEDDINGS2_FILE, 'r', encoding='utf-8') as f:
                quick_log(self.secret_word, f"🔄 Reading JSON file...")
                self.embeddings_cache = json.load(f)
            
            # Signal completion
            loading_complete.set()
            
            quick_log(self.secret_word, f"✅ Loaded embeddings for {len(self.embeddings_cache):,} words")
            
            # Check if secret word has embedding
            if self.secret_word not in self.embeddings_cache:
                quick_log(self.secret_word, f"❌ ERROR: No embedding found for '{self.secret_word}'")
                return False
            
            return True
            
        except Exception as e:
            loading_complete.set()  # Stop status updates
            quick_log(self.secret_word, f"❌ ERROR: Failed to load embeddings: {e}")
            return False
    
    def compute_rankings(self) -> bool:
        """Compute semantic similarity rankings"""
        if not self.embeddings_cache or not self.enable2_words:
            return False
        
        # Provide time estimate for similarity computation
        word_count = len(self.enable2_words)
        if word_count < 10000:
            time_estimate = "30 seconds - 1 minute"
        elif word_count < 50000:
            time_estimate = "2-5 minutes"
        elif word_count < 100000:
            time_estimate = "5-10 minutes"
        else:
            time_estimate = "10-20 minutes"
        
        quick_log(self.secret_word, f"🧮 Computing similarity rankings for {word_count:,} words")
        quick_log(self.secret_word, f"⏰ ESTIMATED TIME: {time_estimate}")
        quick_log(self.secret_word, f"🔄 Progress updates every 30 seconds with ETA...")
        
        # Initialize progress tracker
        self.tracker = create_tracker(self.secret_word, "EMBEDDINGS", len(self.enable2_words))
        
        # Get secret word embedding
        secret_embedding = np.array(self.embeddings_cache[self.secret_word])
        secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)  # Normalize
        
        # Compute similarities
        similarities = []
        words_with_embeddings = []
        
        for i, word in enumerate(self.enable2_words):
            self.tracker.update(i + 1)
            
            if word in self.embeddings_cache:
                # Get and normalize embedding
                word_embedding = np.array(self.embeddings_cache[word])
                word_embedding = word_embedding / np.linalg.norm(word_embedding)
                
                # Compute cosine similarity
                similarity = np.dot(secret_embedding, word_embedding)
                similarities.append(similarity)
                words_with_embeddings.append(word)
            
            # Checkpoint periodically
            if self.tracker.should_checkpoint():
                checkpoint_data = {
                    'processed_words': i + 1,
                    'words_with_embeddings': len(words_with_embeddings)
                }
                self.tracker.checkpoint(checkpoint_data, f"{len(words_with_embeddings):,} embeddings processed")
        
        # Sort by similarity (descending) then alphabetically for ties
        self.tracker.update(len(self.enable2_words), "Sorting by similarity...")
        
        word_similarity_pairs = list(zip(words_with_embeddings, similarities))
        word_similarity_pairs.sort(key=lambda x: (-x[1], x[0]))  # Desc similarity, asc alphabetical
        
        # Create ranked list with ranks
        self.ranked_words = []
        for rank, (word, similarity) in enumerate(word_similarity_pairs, 1):
            self.ranked_words.append((rank, word, similarity))
        
        self.tracker.complete(f"Ranked {len(self.ranked_words):,} words by similarity")
        return True
    
    def save_embeddings_file(self) -> bool:
        """Save ordered embeddings to file"""
        if not self.ranked_words:
            return False
        
        quick_log(self.secret_word, f"💾 Saving ordered embeddings to {self.paths['embeddings']}")
        
        try:
            with open(self.paths['embeddings'], 'w', encoding='utf-8') as f:
                # Header
                f.write("rank,word,similarity\n")
                
                # Write all ranked words
                for rank, word, similarity in self.ranked_words:
                    f.write(f"{rank},{word},{similarity:.8f}\n")
            
            file_size = self.paths['embeddings'].stat().st_size
            quick_log(self.secret_word, f"✅ Saved {len(self.ranked_words):,} ranked words ({file_size/1024/1024:.1f} MB)")
            
            # Show top 10 for verification
            quick_log(self.secret_word, "📊 Top 10 most similar words:")
            for rank, word, similarity in self.ranked_words[:10]:
                quick_log(self.secret_word, f"   {rank:>3}: {word:<15} (similarity: {similarity:.6f})")
            
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to save embeddings file: {e}")
            return False
    
    def generate(self) -> bool:
        """Main generation process"""
        quick_log(self.secret_word, f"🚀 Starting embeddings generation for '{self.secret_word}'")
        
        # Check if file already exists
        if self.paths['embeddings'].exists():
            quick_log(self.secret_word, f"⏭️ Embeddings file already exists: {self.paths['embeddings']}")
            return True
        
        # Step 1: Load word list
        if not self.load_word_list():
            return False
        
        # Step 2: Load embeddings cache
        if not self.load_embeddings_cache():
            return False
        
        # Step 3: Compute rankings
        if not self.compute_rankings():
            return False
        
        # Step 4: Save to file
        if not self.save_embeddings_file():
            return False
        
        quick_log(self.secret_word, f"✅ Embeddings generation completed successfully!")
        return True

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python create_embeddings.py <secret_word>")
        print("Example: python create_embeddings.py forest")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    try:
        generator = EmbeddingsGenerator(secret_word)
        success = generator.generate()
        
        if success:
            print(f"\n🎉 Successfully generated embeddings for '{secret_word}'!")
            sys.exit(0)
        else:
            print(f"\n💥 Failed to generate embeddings for '{secret_word}'")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
