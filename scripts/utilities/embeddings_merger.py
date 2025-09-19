#!/usr/bin/env python3
"""
Embeddings Merger Module
Creates embeddings-[word]2.txt by placing OpenAI words at top and merging with existing rankings
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from .config import Config
    from .progress_tracker import quick_log
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    from progress_tracker import quick_log

class EmbeddingsMerger:
    """Merges OpenAI similar words with existing semantic embeddings"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        
        # Validate word
        valid, result = Config.validate_word(self.secret_word)
        if not valid:
            raise ValueError(f"Invalid secret word: {result}")
        
        self.secret_word = result
        
        # File paths
        self.embeddings_file = Config.SECRETWORD_DIR / f"embeddings-{self.secret_word}.txt"
        self.embeddings2_file = Config.SECRETWORD_DIR / f"embeddings-{self.secret_word}2.txt"
        
        # Data
        self.original_rankings = {}  # word -> (rank, similarity)
        self.openai_words = []       # List of OpenAI words in order
        self.merged_data = []        # Final merged data
    
    def load_original_embeddings(self) -> bool:
        """Load original embeddings file"""
        if not self.embeddings_file.exists():
            quick_log(self.secret_word, f"❌ ERROR: Original embeddings file not found: {self.embeddings_file}")
            return False
        
        quick_log(self.secret_word, f"📂 Loading original embeddings from {self.embeddings_file}")
        
        try:
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        rank = int(row[0])
                        word = row[1].lower()
                        similarity = float(row[2])
                        self.original_rankings[word] = (rank, similarity)
            
            quick_log(self.secret_word, f"✅ Loaded {len(self.original_rankings):,} original rankings")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to load original embeddings: {e}")
            return False
    
    def merge_rankings(self, openai_words: List[str]) -> bool:
        """Merge OpenAI words with original rankings"""
        if not self.original_rankings:
            return False
        
        self.openai_words = openai_words
        quick_log(self.secret_word, f"🔄 Merging {len(openai_words)} OpenAI words with original rankings...")
        
        # Track statistics
        found_in_original = 0
        not_found_in_original = 0
        moved_from_original = 0
        
        # Create merged data list
        self.merged_data = []
        used_words = set()  # Track words we've already placed
        
        # Phase 1: Add OpenAI words at the top
        for openai_rank, word in enumerate(openai_words, 1):
            word_lower = word.lower()
            
            if word_lower in self.original_rankings:
                # Word exists in original - use its similarity but new rank
                original_rank, similarity = self.original_rankings[word_lower]
                note = f"OpenAI #{openai_rank}, Our rank #{original_rank:,}"
                found_in_original += 1
                
                if original_rank > openai_rank:
                    moved_from_original += 1
            else:
                # Word not in original - use placeholder similarity
                similarity = 0.0  # Will be obvious it's not from embeddings
                note = f"OpenAI #{openai_rank}, Not in our embeddings"
                not_found_in_original += 1
            
            self.merged_data.append({
                'rank': openai_rank,
                'word': word_lower,
                'similarity': similarity,
                'note': note
            })
            
            used_words.add(word_lower)
        
        # Phase 2: Add remaining words from original rankings
        remaining_words = []
        for word, (original_rank, similarity) in self.original_rankings.items():
            if word not in used_words:
                remaining_words.append((original_rank, word, similarity))
        
        # Sort remaining words by their original rank
        remaining_words.sort(key=lambda x: x[0])
        
        # Add remaining words after OpenAI words
        next_rank = len(openai_words) + 1
        for original_rank, word, similarity in remaining_words:
            self.merged_data.append({
                'rank': next_rank,
                'word': word,
                'similarity': similarity,
                'note': f"Our original rank #{original_rank:,}"
            })
            next_rank += 1
        
        # Report statistics
        quick_log(self.secret_word, f"✅ Merge completed:")
        quick_log(self.secret_word, f"   OpenAI words found in original: {found_in_original}/{len(openai_words)} ({found_in_original/len(openai_words)*100:.1f}%)")
        quick_log(self.secret_word, f"   OpenAI words not in original: {not_found_in_original}")
        quick_log(self.secret_word, f"   Words moved up from original position: {moved_from_original}")
        quick_log(self.secret_word, f"   Total merged entries: {len(self.merged_data):,}")
        
        return True
    
    def save_merged_embeddings(self) -> bool:
        """Save merged embeddings to embeddings2 file"""
        if not self.merged_data:
            return False
        
        quick_log(self.secret_word, f"💾 Saving merged embeddings to {self.embeddings2_file}")
        
        try:
            Config.ensure_directories()
            
            with open(self.embeddings2_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['rank', 'word', 'similarity', 'note'])
                
                # Write all merged data
                for entry in self.merged_data:
                    writer.writerow([
                        entry['rank'],
                        entry['word'],
                        f"{entry['similarity']:.8f}",
                        entry['note']
                    ])
            
            file_size = self.embeddings2_file.stat().st_size
            quick_log(self.secret_word, f"✅ Saved merged embeddings ({file_size/1024/1024:.1f} MB)")
            
            # Show top 10 for verification
            quick_log(self.secret_word, "📊 Top 10 merged rankings:")
            for i, entry in enumerate(self.merged_data[:10]):
                quick_log(self.secret_word, f"   {entry['rank']:>3}: {entry['word']:<15} ({entry['note']})")
            
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to save merged embeddings: {e}")
            return False
    
    def create_merged_embeddings(self, openai_words: List[str]) -> bool:
        """Main process to create merged embeddings file"""
        quick_log(self.secret_word, f"🚀 Creating merged embeddings for '{self.secret_word}'")
        
        # Check if merged file already exists
        if self.embeddings2_file.exists():
            quick_log(self.secret_word, f"⏭️ Merged embeddings file already exists: {self.embeddings2_file}")
            return True
        
        # Step 1: Load original embeddings
        if not self.load_original_embeddings():
            return False
        
        # Step 2: Merge with OpenAI words
        if not self.merge_rankings(openai_words):
            return False
        
        # Step 3: Save merged file
        if not self.save_merged_embeddings():
            return False
        
        quick_log(self.secret_word, f"✅ Merged embeddings creation completed successfully!")
        return True

def create_merged_embeddings(secret_word: str, openai_words: List[str]) -> bool:
    """Convenience function to create merged embeddings"""
    try:
        merger = EmbeddingsMerger(secret_word)
        return merger.create_merged_embeddings(openai_words)
    except Exception as e:
        quick_log(secret_word, f"❌ ERROR in embeddings merger: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python embeddings_merger.py <secret_word>")
        print("Example: python embeddings_merger.py forest")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    # For testing, create some dummy OpenAI words
    test_openai_words = [
        "woods", "woodland", "trees", "jungle", "wilderness",
        "timber", "grove", "thicket", "vegetation", "nature"
    ]
    
    try:
        merger = EmbeddingsMerger(secret_word)
        success = merger.create_merged_embeddings(test_openai_words)
        
        if success:
            print(f"\n🎉 Successfully created merged embeddings for '{secret_word}'!")
        else:
            print(f"\n💥 Failed to create merged embeddings for '{secret_word}'")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)
