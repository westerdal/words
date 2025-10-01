#!/usr/bin/env python3
"""
Batch Embeddings Generator for Multiple Secret Words
Efficiently processes 100+ secret words by loading embeddings once.
"""

import json
import sys
import time
import argparse
import psutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utilities.progress_tracker import quick_log

class BatchEmbeddingsGenerator:
    def __init__(self):
        self.embeddings = None
        self.words = None
        self.start_time = None
        self.processed_count = 0
        self.total_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        used_gb = memory_info.rss / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        return used_gb, available_gb
    
    def memory_check(self):
        """Monitor memory usage and warn if approaching limits"""
        used_gb, available_gb = self.get_memory_usage()
        total_gb = psutil.virtual_memory().total / (1024**3)
        usage_percent = (used_gb / total_gb) * 100
        
        if usage_percent > 80:
            print(f"⚠️  WARNING: High memory usage: {used_gb:.1f} GB ({usage_percent:.1f}%)")
        
        return used_gb, available_gb
    
    def load_resources_once(self):
        """Load embeddings2.json and enable2.txt once into memory"""
        print(f"🚀 Starting batch embeddings generation")
        self.start_time = time.time()
        
        # Check initial memory
        used_gb, available_gb = self.memory_check()
        print(f"💾 Initial memory usage: {used_gb:.1f} GB / {available_gb + used_gb:.1f} GB available")
        
        # Load embeddings
        embeddings_path = Path(".venv/embeddings2.json")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        file_size_gb = embeddings_path.stat().st_size / (1024**3)
        print(f"📂 Loading embeddings2.json ({file_size_gb:.1f} GB)...")
        print(f"⏰ ESTIMATED TIME: 5-10 minutes")
        
        load_start = time.time()
        with open(embeddings_path, 'r') as f:
            self.embeddings = json.load(f)
        load_time = time.time() - load_start
        
        print(f"✅ Loaded {len(self.embeddings):,} embeddings in {load_time/60:.1f} minutes")
        
        # Load word list
        words_path = Path("data/enable2.txt")
        with open(words_path, 'r') as f:
            self.words = [line.strip() for line in f if line.strip()]
        
        print(f"📂 Loaded {len(self.words):,} words from ENABLE2")
        
        # Check memory after loading
        used_gb, available_gb = self.memory_check()
        print(f"💾 Total memory usage after loading: {used_gb:.1f} GB / {available_gb + used_gb:.1f} GB")
        
        if available_gb < 2.0:
            print("⚠️  WARNING: Low available memory. Consider processing smaller batches.")
    
    def process_word(self, secret_word: str) -> List[Tuple[int, float, str]]:
        """Process single word using in-memory embeddings"""
        if secret_word not in self.embeddings:
            raise ValueError(f"No embedding found for '{secret_word}'")
        
        secret_embedding = np.array(self.embeddings[secret_word]).reshape(1, -1)
        rankings = []
        
        for word in self.words:
            if word in self.embeddings:
                word_embedding = np.array(self.embeddings[word]).reshape(1, -1)
                similarity = cosine_similarity(secret_embedding, word_embedding)[0][0]
                rankings.append((similarity, word))
        
        # Sort by similarity (descending), then alphabetically for ties
        rankings.sort(key=lambda x: (-x[0], x[1]))
        
        # Convert to rank, similarity, word format
        result = []
        for rank, (similarity, word) in enumerate(rankings, 1):
            result.append((rank, similarity, word))
        
        return result
    
    def save_embeddings_file(self, word: str, rankings: List[Tuple[int, float, str]]):
        """Save individual embeddings-[word].txt file"""
        output_path = Path(f"secretword/embeddings-{word}.txt")
        
        with open(output_path, 'w') as f:
            for rank, similarity, ranked_word in rankings:
                f.write(f"{rank} {similarity:.6f} {ranked_word}\n")
        
        file_size_mb = output_path.stat().st_size / (1024**2)
        return file_size_mb
    
    def calculate_eta(self, processed: int, total: int) -> str:
        """Calculate estimated time remaining for large batches"""
        if processed == 0:
            return "calculating..."
        
        elapsed = time.time() - self.start_time
        avg_time_per_word = elapsed / processed
        remaining_words = total - processed
        eta_seconds = remaining_words * avg_time_per_word
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def process_batch(self, secret_words: List[str], overwrite: bool = False):
        """Process multiple words efficiently with progress tracking"""
        self.total_count = len(secret_words)
        self.processed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        
        print(f"🔄 Processing batch: {self.total_count} words")
        
        failed_words = []
        
        for i, word in enumerate(secret_words, 1):
            try:
                # Check if file already exists
                output_path = Path(f"secretword/embeddings-{word}.txt")
                if output_path.exists() and not overwrite:
                    print(f"⏭️  #{i}/{self.total_count} | {word} | already exists, skipping")
                    self.skipped_count += 1
                    continue
                
                # Process the word
                word_start = time.time()
                rankings = self.process_word(word)
                file_size_mb = self.save_embeddings_file(word, rankings)
                word_time = time.time() - word_start
                
                self.processed_count += 1
                eta = self.calculate_eta(i, self.total_count)
                
                print(f"✅ #{i}/{self.total_count} | {word} | embeddings-{word}.txt saved ({file_size_mb:.1f}MB) | ETA: {eta}")
                
                # Progress milestones for large batches
                if i % 10 == 0 or i in [25, 50, 75]:
                    progress_pct = (i / self.total_count) * 100
                    print(f"📊 Progress: {progress_pct:.0f}% complete | ETA: {eta}")
                    self.memory_check()
                
            except Exception as e:
                print(f"❌ #{i}/{self.total_count} | {word} | ERROR: {str(e)}")
                failed_words.append(word)
                self.failed_count += 1
                continue
        
        # Final summary
        total_time = time.time() - self.start_time
        print(f"\n🎉 Batch complete! Processed in {total_time/60:.0f} minutes")
        print(f"📊 Results: {self.processed_count} processed, {self.skipped_count} skipped, {self.failed_count} failed")
        
        if failed_words:
            print(f"❌ Failed words: {', '.join(failed_words)}")
        
        # Calculate time savings
        serial_time = self.total_count * 9  # 7min load + 2min process per word
        time_saved = serial_time - (total_time/60)
        if time_saved > 0:
            print(f"💰 Time saved: {time_saved:.0f} minutes ({time_saved/serial_time*100:.0f}% faster than serial processing)")

def get_words_from_master_list(count: int = None, start_idx: int = 0) -> List[str]:
    """Get words from master-list.txt"""
    master_list_path = Path("secretword/master-list.txt")
    if not master_list_path.exists():
        raise FileNotFoundError("master-list.txt not found in secretword/")
    
    with open(master_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Extract just the word part from "difficulty-category-word" format
    words = []
    for line in lines[start_idx:]:
        parts = line.split('-')
        if len(parts) >= 3:
            word = parts[-1]  # Last part is the word
            words.append(word)
    
    if count:
        words = words[:count]
    
    return words

def main():
    parser = argparse.ArgumentParser(description="Batch process embeddings for multiple secret words")
    parser.add_argument('--count', type=int, help='Number of words to process from master-list.txt')
    parser.add_argument('--start', type=int, default=0, help='Starting index in master-list.txt')
    parser.add_argument('--words', nargs='*', help='Specific words to process')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing embedding files')
    parser.add_argument('--all-remaining', action='store_true', help='Process all remaining words from master-list.txt')
    
    args = parser.parse_args()
    
    try:
        # Determine which words to process
        if args.words:
            secret_words = args.words
        elif args.all_remaining:
            secret_words = get_words_from_master_list(start_idx=args.start)
        elif args.count:
            secret_words = get_words_from_master_list(count=args.count, start_idx=args.start)
        else:
            # Default: process next 10 words
            secret_words = get_words_from_master_list(count=10, start_idx=args.start)
        
        if not secret_words:
            print("❌ No words to process")
            return
        
        print(f"🎯 Words to process: {', '.join(secret_words[:10])}" + (f" ... and {len(secret_words)-10} more" if len(secret_words) > 10 else ""))
        
        # Create batch processor
        generator = BatchEmbeddingsGenerator()
        
        # Load resources once
        generator.load_resources_once()
        
        # Process all words
        generator.process_batch(secret_words, overwrite=args.overwrite)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        if hasattr(generator, 'processed_count'):
            print(f"📊 Progress before interruption: {generator.processed_count} words processed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


