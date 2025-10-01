#!/usr/bin/env python3
"""
Generate complete CSV for a secret word with AI clues and dynamic cutoff
Usage: python generate_csv.py <secret_word>
"""

import json
import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import openai

# Add utilities to path
sys.path.append(str(Path(__file__).parent.parent / "utilities"))
from config import Config, CONNECTION_STRENGTHS, SPECIAL_CLUES
from progress_tracker import create_tracker, quick_log
from clue_cache import ClueCache

class CSVGenerator:
    """Generates complete CSV for a secret word with AI clues and dynamic cutoff"""
    
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
        self.ranked_words = []  # (rank, word, similarity)
        self.csv_data = []      # Final CSV rows
        
        # AI tracking
        self.ai_cutoff_reached = False
        self.ai_calls_made = 0
        self.total_words_processed = 0  # Track ALL words encountered
        
        # Initialize OpenAI
        if Config.check_openai_key():
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.ai_available = True
        else:
            self.ai_available = False
            quick_log(self.secret_word, "⚠️ WARNING: OpenAI API key not available - will use NULL clues")
    
    def load_embeddings_file(self) -> bool:
        """Load ranked words from embeddings file (prefer clean versions to avoid secret word contamination)"""
        # Priority: 1) Clean enhanced, 2) Clean standard, 3) Original enhanced, 4) Original standard
        embeddings_file = self.paths['embeddings']
        
        # Check for enhanced version (with "2" suffix)
        embeddings2_file = embeddings_file.parent / f"{self.secret_word}2-embeddings.txt"
        
        if embeddings2_file.exists():
            target_file = embeddings2_file
            quick_log(self.secret_word, f"📂 Using enhanced embeddings (with improved AI prompt): {target_file}")
        elif embeddings_file.exists():
            target_file = embeddings_file
            quick_log(self.secret_word, f"📂 Using standard embeddings (with improved AI prompt): {target_file}")
        else:
            quick_log(self.secret_word, f"❌ ERROR: No embeddings file found")
            quick_log(self.secret_word, f"   Looked for: {embeddings_file}")
            return False
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                quick_log(self.secret_word, f"❌ ERROR: Embeddings file is empty")
                return False
            
            # Detect format by checking first line
            first_line = lines[0].strip()
            
            if first_line.startswith('rank,word,similarity') or ',' in first_line:
                # New CSV format
                quick_log(self.secret_word, f"📋 Detected new CSV format")
                reader = csv.reader(lines)
                if first_line.startswith('rank,word,similarity'):
                    next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        rank = int(row[0])
                        word = row[1]
                        similarity = float(row[2])
                        self.ranked_words.append((rank, word, similarity))
            
            else:
                # Old space-separated format: "1 1.000000 rock"
                quick_log(self.secret_word, f"📋 Detected old space-separated format")
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            rank = int(parts[0])
                            similarity = float(parts[1])
                            word = parts[2]
                            self.ranked_words.append((rank, word, similarity))
            
            quick_log(self.secret_word, f"✅ Loaded {len(self.ranked_words):,} ranked words")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to load embeddings file: {e}")
            return False
    
    def get_ai_clues_batch(self, words_with_ranks: List[tuple]) -> Dict[str, Dict[str, str]]:
        """Get AI clues and strength assessments for a batch of words with their ranks"""
        if not self.ai_available or self.ai_cutoff_reached:
            return {word: {'clue': None, 'strength': 'hard_cutoff'} for word, rank in words_with_ranks}
        
        # Check cache first
        cached_results, api_needed = self.cache.lookup_batch(words_with_ranks)
        
        # If all words are cached, return immediately
        if not api_needed:
            return cached_results
        
        try:
            # Import the standalone clue generation function
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from generate_clue import generate_clue
            
            # Extract just the words for clue generation
            words_to_process = [word for word, rank in api_needed]
            
            # Generate clues using the standalone function
            clues_data = generate_clue(
                secret_word=self.secret_word,
                guess_words=words_to_process,
                model=Config.OPENAI_MODEL,
                temperature=0.7,
                max_tokens=2000
            )
            
            self.ai_calls_made += 1
            
            # Validate and clean API results
            api_results = {}
            word_rank_map = {word: rank for word, rank in api_needed}
            for word, rank in api_needed:
                if word in clues_data:
                    data = clues_data[word]
                    clue = data.get('clue', 'Super close, sizzling hot')
                    strength = data.get('strength', 'medium').lower()
                    
                    # Validate strength
                    if strength not in ['strong', 'medium', 'weak']:
                        strength = 'medium'
                    
                    # Ensure clue is string
                    if not isinstance(clue, str):
                        clue = 'Super close, sizzling hot'
                    
                    # CRITICAL: Check if clue contains the secret word
                    if self.secret_word.lower() in clue.lower():
                        quick_log(self.secret_word, f"🔥 #{rank:,} | {word:15} | 'Super close, sizzling hot' | {self.total_words_processed:,}/{Config.HARD_CUTOFF_RANK:,} processed")
                        clue = 'Super close, sizzling hot'
                        strength = 'strong'  # These are actually very close relationships
                    else:
                        quick_log(self.secret_word, f"✅ #{rank:,} | {word:15} | '{clue[:25]}{'...' if len(clue) > 25 else ''}' | {self.total_words_processed:,}/{Config.HARD_CUTOFF_RANK:,} processed")
                    
                    api_results[word] = {'clue': clue, 'strength': strength}
                else:
                    api_results[word] = {'clue': 'Super close, sizzling hot', 'strength': 'medium'}
                
            # Store API results in cache
            if api_results:
                self.cache.store_batch(api_results, api_needed)
            
            # Combine cached and API results
            final_results = {**cached_results, **api_results}
            return final_results
                
        except Exception as e:
            quick_log(self.secret_word, f"⚠️ WARNING: AI API call failed: {e}")
            fallback_results = {word: {'clue': 'Super close, sizzling hot', 'strength': 'medium'} for word, rank in api_needed}
            return {**cached_results, **fallback_results}
    
    def process_word(self, rank: int, word: str, similarity: float) -> Dict[str, Any]:
        """Process a single word and return CSV row data"""
        # Handle secret word specially
        if word == self.secret_word:
            return {
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': SPECIAL_CLUES['secret_word'],
                'connection_strength': CONNECTION_STRENGTHS['secret_word']
            }
        
        # Check if we should use AI
        if self.ai_cutoff_reached or rank > Config.HARD_CUTOFF_RANK or self.total_words_processed >= Config.MIN_AI_CLUES:
            # Log when we hit the hard word limit
            if self.total_words_processed >= Config.MIN_AI_CLUES and not self.ai_cutoff_reached:
                quick_log(self.secret_word, f"🎯 Minimum AI clues reached - processed {Config.MIN_AI_CLUES:,} words - switching to NULL clues for remaining words")
                self.ai_cutoff_reached = True
            return {
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': None,
                'connection_strength': CONNECTION_STRENGTHS['hard_cutoff']
            }
        
        # Get AI clue
        ai_result = self.get_ai_clues_batch([(word, rank)])
        word_data = ai_result[word]
        
        clue = word_data['clue']
        strength = word_data['strength']
        
        return {
            'rank': rank,
            'secret_word': self.secret_word,
            'word': word,
            'clue': clue,
            'connection_strength': strength
        }
    
    def process_all_words(self) -> bool:
        """Process all words and generate CSV data"""
        if not self.ranked_words:
            return False
        
        # Initialize progress tracker and cache
        self.tracker = create_tracker(self.secret_word, "CSV_GENERATION", len(self.ranked_words))
        self.cache = ClueCache(self.secret_word)
        
        # Process words in batches for efficiency
        batch_size = Config.AI_BATCH_SIZE
        current_batch = []
        processed_count = 0
        
        for rank, word, similarity in self.ranked_words:
            processed_count += 1
            self.total_words_processed += 1  # Track at class level
            self.tracker.update(processed_count)
            
            # Handle secret word specially
            if word == self.secret_word:
                row_data = {
                    'rank': rank,
                    'secret_word': self.secret_word,
                    'word': word,
                    'clue': SPECIAL_CLUES['secret_word'],
                    'connection_strength': CONNECTION_STRENGTHS['secret_word']
                }
                self.csv_data.append(row_data)
                continue
            
            # Check if AI cutoff reached, hard cutoff by rank, or processed max words
            if self.ai_cutoff_reached or rank > Config.HARD_CUTOFF_RANK or self.total_words_processed >= Config.HARD_CUTOFF_RANK:
                # Log when we hit the hard word limit
                if self.total_words_processed >= Config.HARD_CUTOFF_RANK and not self.ai_cutoff_reached:
                    quick_log(self.secret_word, f"🎯 Hard cutoff reached - processed {Config.HARD_CUTOFF_RANK:,} words - switching to NULL clues for remaining words")
                    self.ai_cutoff_reached = True  # Set flag to avoid repeated messages
                
                row_data = {
                    'rank': rank,
                    'secret_word': self.secret_word,
                    'word': word,
                    'clue': None,
                    'connection_strength': CONNECTION_STRENGTHS['hard_cutoff']
                }
                self.csv_data.append(row_data)
                continue
            
            # Add to batch for AI processing
            current_batch.append((rank, word, similarity))
            
            # Process batch when full or AI not available
            if len(current_batch) >= batch_size or not self.ai_available:
                self._process_batch(current_batch)
                current_batch = []
            
            # Checkpoint periodically
            if self.tracker.should_checkpoint():
                checkpoint_data = {
                    'processed_count': processed_count,
                    'ai_calls_made': self.ai_calls_made,
                    'ai_cutoff_reached': self.ai_cutoff_reached,
                    'csv_rows': len(self.csv_data)
                }
                self.tracker.checkpoint(checkpoint_data, f"{len(self.csv_data):,} rows generated")
        
        # Process remaining batch
        if current_batch:
            self._process_batch(current_batch)
        
        self.tracker.complete(f"Generated {len(self.csv_data):,} CSV rows")
        return True
    
    def _process_batch(self, batch: List[Tuple[int, str, float]]):
        """Process a batch of words"""
        if not batch:
            return
        
        if not self.ai_available or self.ai_cutoff_reached:
            # No AI - just add with NULL clues
            for rank, word, similarity in batch:
                row_data = {
                    'rank': rank,
                    'secret_word': self.secret_word,
                    'word': word,
                    'clue': None,
                    'connection_strength': CONNECTION_STRENGTHS['hard_cutoff']
                }
                self.csv_data.append(row_data)
            return
        
        # Extract words with ranks for AI call
        words_with_ranks = [(word, rank) for rank, word, similarity in batch]
        
        # Get AI results
        ai_results = self.get_ai_clues_batch(words_with_ranks)
        
        # Process results
        for rank, word, similarity in batch:
            if word in ai_results:
                word_data = ai_results[word]
                clue = word_data['clue']
                strength = word_data['strength']
                
                # No special handling for weak connections - just add to CSV
                
                row_data = {
                    'rank': rank,
                    'secret_word': self.secret_word,
                    'word': word,
                    'clue': clue,
                    'connection_strength': strength
                }
                self.csv_data.append(row_data)
            else:
                # Fallback
                row_data = {
                    'rank': rank,
                    'secret_word': self.secret_word,
                    'word': word,
                    'clue': 'Super close, sizzling hot',
                    'connection_strength': 'medium'
                }
                self.csv_data.append(row_data)
    
    
    def save_csv(self) -> bool:
        """Save final CSV file"""
        if not self.csv_data:
            return False
        
        csv_file = self.paths['csv']
        quick_log(self.secret_word, f"💾 Saving CSV to {csv_file}")
        
        try:
            # Sort by rank
            self.csv_data.sort(key=lambda x: x['rank'])
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=Config.CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self.csv_data)
            
            # Report statistics
            file_size = csv_file.stat().st_size
            ai_clues = len([row for row in self.csv_data if row['clue'] and row['clue'] not in [SPECIAL_CLUES['secret_word'], 'ERROR'] and row['clue'] is not None])
            null_clues = len([row for row in self.csv_data if row['clue'] is None])
            
            quick_log(self.secret_word, f"✅ Saved {len(self.csv_data):,} rows ({file_size/1024/1024:.1f} MB)")
            quick_log(self.secret_word, f"📊 AI clues: {ai_clues:,} | NULL clues: {null_clues:,}")
            
            # Show cache statistics
            cache_stats = self.cache.get_stats()
            quick_log(self.secret_word, f"💾 Cache: {cache_stats['session_hits']:,} hits | {cache_stats['session_misses']:,} misses | {cache_stats['hit_rate']:.1%} hit rate")
            if cache_stats['estimated_time_saved_minutes'] > 0:
                quick_log(self.secret_word, f"⚡ Time saved: {cache_stats['estimated_time_saved_minutes']:.1f} min | Cost saved: ${cache_stats['estimated_cost_saved_usd']:.2f}")
            
            # Save cache before finishing
            self.cache.save_cache()
            
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to save CSV: {e}")
            return False
    
    def generate(self) -> bool:
        """Main generation process"""
        quick_log(self.secret_word, f"🚀 Starting CSV generation for '{self.secret_word}'")
        
        # Check if CSV already exists
        if self.paths['csv'].exists():
            quick_log(self.secret_word, f"⏭️ CSV file already exists: {self.paths['csv']}")
            return True
        
        # Step 1: Load embeddings file
        if not self.load_embeddings_file():
            return False
        
        # Step 2: Process all words
        if not self.process_all_words():
            return False
        
        # Step 3: Save CSV
        if not self.save_csv():
            return False
        
        quick_log(self.secret_word, f"✅ CSV generation completed successfully!")
        return True

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python generate_csv.py <secret_word>")
        print("Example: python generate_csv.py forest")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    # Validate word
    valid, result = Config.validate_word(secret_word)
    if not valid:
        print(f"Error: {result}")
        sys.exit(1)
    
    secret_word = result
    
    try:
        generator = CSVGenerator(secret_word)
        success = generator.generate()
        
        if success:
            print(f"\n🎉 Successfully generated CSV for '{secret_word}'!")
            sys.exit(0)
        else:
            print(f"\n💥 Failed to generate CSV for '{secret_word}'")
            sys.exit(1)
            
    except Exception as e:
        try:
            print(f"\n💥 Error: {e}")
        except UnicodeEncodeError:
            print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
