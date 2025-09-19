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
        self.weak_queue = []    # Weak connections to reinsert
        
        # AI tracking
        self.consecutive_weak = 0
        self.ai_cutoff_reached = False
        self.ai_calls_made = 0
        
        # Initialize OpenAI
        if Config.check_openai_key():
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.ai_available = True
        else:
            self.ai_available = False
            quick_log(self.secret_word, "⚠️ WARNING: OpenAI API key not available - will use NULL clues")
    
    def load_embeddings_file(self) -> bool:
        """Load ranked words from embeddings file (prefer embeddings2 if available)"""
        # Try enhanced embeddings2 file first
        embeddings2_file = Config.SECRETWORD_DIR / f"embeddings-{self.secret_word}2.txt"
        embeddings_file = self.paths['embeddings']
        
        if embeddings2_file.exists():
            target_file = embeddings2_file
            quick_log(self.secret_word, f"📂 Using enhanced embeddings file: {target_file}")
        elif embeddings_file.exists():
            target_file = embeddings_file
            quick_log(self.secret_word, f"📂 Using standard embeddings file: {target_file}")
        else:
            quick_log(self.secret_word, f"❌ ERROR: No embeddings file found")
            quick_log(self.secret_word, f"   Looked for: {embeddings2_file}")
            quick_log(self.secret_word, f"   Looked for: {embeddings_file}")
            return False
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        rank = int(row[0])
                        word = row[1]
                        similarity = float(row[2])
                        self.ranked_words.append((rank, word, similarity))
            
            quick_log(self.secret_word, f"✅ Loaded {len(self.ranked_words):,} ranked words")
            return True
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ ERROR: Failed to load embeddings file: {e}")
            return False
    
    def get_ai_clues_batch(self, words_batch: List[str]) -> Dict[str, Dict[str, str]]:
        """Get AI clues and strength assessments for a batch of words"""
        if not self.ai_available or self.ai_cutoff_reached:
            return {word: {'clue': None, 'strength': 'hard_cutoff'} for word in words_batch}
        
        try:
            # Create prompt
            prompt = (
                f"For each word below, describe the SPECIFIC relationship between the word and '{self.secret_word}' "
                f"in 7 words or less. Focus on HOW they connect, not what the word is.\n\n"
                f"CRITICAL: NEVER use the word '{self.secret_word}' in any clue. "
                f"Instead use generic terms like 'that field', 'that medium', 'that form', 'that concept', 'that area'.\n\n"
                f"Also assess the connection strength as 'strong', 'medium', or 'weak'.\n\n"
                f"Return JSON format:\n"
                f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
                f"Words: {words_batch}"
            )
            
            # Make API call
            response = openai.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            
            self.ai_calls_made += 1
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON
            try:
                # Find JSON block
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    clues_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON block found")
                
                # Validate and clean results
                results = {}
                for word in words_batch:
                    if word in clues_data:
                        data = clues_data[word]
                        clue = data.get('clue', 'ERROR')
                        strength = data.get('strength', 'medium').lower()
                        
                        # Validate strength
                        if strength not in ['strong', 'medium', 'weak']:
                            strength = 'medium'
                        
                        # Ensure clue is string
                        if not isinstance(clue, str):
                            clue = 'ERROR'
                        
                        # CRITICAL: Check if clue contains the secret word
                        if self.secret_word.lower() in clue.lower():
                            quick_log(self.secret_word, f"⚠️ REJECTED clue for '{word}': contains secret word - '{clue}'")
                            clue = 'ERROR'
                            strength = 'weak'
                        
                        results[word] = {'clue': clue, 'strength': strength}
                    else:
                        results[word] = {'clue': 'ERROR', 'strength': 'medium'}
                
                return results
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                quick_log(self.secret_word, f"⚠️ WARNING: Failed to parse AI response: {e}")
                return {word: {'clue': 'ERROR', 'strength': 'medium'} for word in words_batch}
                
        except Exception as e:
            quick_log(self.secret_word, f"⚠️ WARNING: AI API call failed: {e}")
            return {word: {'clue': 'ERROR', 'strength': 'medium'} for word in words_batch}
    
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
        if self.ai_cutoff_reached or rank > Config.HARD_CUTOFF_RANK:
            return {
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': None,
                'connection_strength': CONNECTION_STRENGTHS['hard_cutoff']
            }
        
        # Get AI clue
        ai_result = self.get_ai_clues_batch([word])
        word_data = ai_result[word]
        
        clue = word_data['clue']
        strength = word_data['strength']
        
        # Track consecutive weak connections
        if strength == 'weak':
            self.consecutive_weak += 1
            # Only allow cutoff if we've generated minimum required clues
            if (self.consecutive_weak >= Config.WEAK_CONNECTION_THRESHOLD and 
                len(self.csv_data) >= Config.MIN_AI_CLUES):
                self.ai_cutoff_reached = True
                quick_log(self.secret_word, f"🛑 Dynamic cutoff reached after {self.consecutive_weak} consecutive weak connections at rank {rank} (generated {len(self.csv_data)} clues)")
            elif self.consecutive_weak >= Config.WEAK_CONNECTION_THRESHOLD:
                quick_log(self.secret_word, f"⚠️ Would cutoff at {self.consecutive_weak} weak connections, but only have {len(self.csv_data)} clues (need {Config.MIN_AI_CLUES})")
        else:
            self.consecutive_weak = 0
        
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
        
        # Initialize progress tracker
        self.tracker = create_tracker(self.secret_word, "CSV_GENERATION", len(self.ranked_words))
        
        # Process words in batches for efficiency
        batch_size = Config.AI_BATCH_SIZE
        current_batch = []
        processed_count = 0
        
        for rank, word, similarity in self.ranked_words:
            processed_count += 1
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
            
            # Check if AI cutoff reached or hard cutoff
            if self.ai_cutoff_reached or rank > Config.HARD_CUTOFF_RANK:
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
                    'consecutive_weak': self.consecutive_weak,
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
        
        # Extract words for AI call
        words = [word for rank, word, similarity in batch]
        
        # Get AI results
        ai_results = self.get_ai_clues_batch(words)
        
        # Process results
        for rank, word, similarity in batch:
            if word in ai_results:
                word_data = ai_results[word]
                clue = word_data['clue']
                strength = word_data['strength']
                
                # Track consecutive weak connections
                if strength == 'weak':
                    self.consecutive_weak += 1
                    # Only allow cutoff if we've generated minimum required clues
                    if (self.consecutive_weak >= Config.WEAK_CONNECTION_THRESHOLD and 
                        len(self.csv_data) >= Config.MIN_AI_CLUES):
                        self.ai_cutoff_reached = True
                        quick_log(self.secret_word, f"🛑 Dynamic cutoff reached after {self.consecutive_weak} consecutive weak connections at rank {rank} (generated {len(self.csv_data)} clues)")
                    elif self.consecutive_weak >= Config.WEAK_CONNECTION_THRESHOLD:
                        quick_log(self.secret_word, f"⚠️ Would cutoff at {self.consecutive_weak} weak connections, but only have {len(self.csv_data)} clues (need {Config.MIN_AI_CLUES})")
                else:
                    self.consecutive_weak = 0
                
                # Check if this is a weak connection to queue
                if strength == 'weak' and not self.ai_cutoff_reached:
                    # Add to weak queue instead of main CSV
                    weak_entry = {
                        'rank': rank,
                        'word': word,
                        'similarity': similarity,
                        'clue': 'weak connection',
                        'strength': 'weak'
                    }
                    self.weak_queue.append(weak_entry)
                    continue
                
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
                    'clue': 'ERROR',
                    'connection_strength': 'medium'
                }
                self.csv_data.append(row_data)
    
    def handle_weak_queue(self):
        """Handle weak connection queuing and reinsertion"""
        if not self.weak_queue:
            return
        
        quick_log(self.secret_word, f"🗂️ Processing {len(self.weak_queue)} weak connections from queue")
        
        # Save weak queue to file
        queue_data = {
            'secret_word': self.secret_word,
            'weak_connections': self.weak_queue,
            'total_count': len(self.weak_queue)
        }
        
        try:
            with open(self.paths['queue'], 'w', encoding='utf-8') as f:
                json.dump(queue_data, f, indent=2)
            quick_log(self.secret_word, f"💾 Saved weak queue to {self.paths['queue']}")
        except Exception as e:
            quick_log(self.secret_word, f"⚠️ WARNING: Failed to save weak queue: {e}")
        
        # Remove weak connections from main CSV (they were not added due to queuing)
        # Resequence ranks
        self.csv_data.sort(key=lambda x: x['rank'])
        for i, row in enumerate(self.csv_data):
            row['rank'] = i + 1
        
        # Add weak connections back at rank 10,000+
        start_rank = max(10000, len(self.csv_data) + 1)
        
        for i, weak_entry in enumerate(self.weak_queue):
            row_data = {
                'rank': start_rank + i,
                'secret_word': self.secret_word,
                'word': weak_entry['word'],
                'clue': 'weak connection',
                'connection_strength': 'weak'
            }
            self.csv_data.append(row_data)
        
        quick_log(self.secret_word, f"✅ Reinserted {len(self.weak_queue)} weak connections at ranks {start_rank}+")
    
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
            ai_clues = len([row for row in self.csv_data if row['clue'] and row['clue'] not in [SPECIAL_CLUES['secret_word'], 'ERROR', 'weak connection'] and row['clue'] is not None])
            weak_connections = len([row for row in self.csv_data if row['connection_strength'] == 'weak'])
            null_clues = len([row for row in self.csv_data if row['clue'] is None])
            
            quick_log(self.secret_word, f"✅ Saved {len(self.csv_data):,} rows ({file_size/1024/1024:.1f} MB)")
            quick_log(self.secret_word, f"📊 AI clues: {ai_clues:,} | Weak connections: {weak_connections:,} | NULL clues: {null_clues:,}")
            
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
        
        # Step 3: Handle weak queue
        self.handle_weak_queue()
        
        # Step 4: Save CSV
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
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
