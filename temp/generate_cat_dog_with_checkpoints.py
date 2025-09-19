#!/usr/bin/env python3
"""
Generate cat and dog CSV files with dynamic cutoff and checkpoint saving
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
from datetime import datetime

class CheckpointSemanticGenerator:
    """Semantic generator with checkpoint saving capability"""
    
    def __init__(self, secret_word, consecutive_weak_threshold=5):
        self.secret_word = secret_word.lower()
        self.consecutive_weak_threshold = consecutive_weak_threshold
        self.consecutive_weak_count = 0
        self.ai_cutoff_reached = False
        self.cutoff_rank = None
        self.total_ai_calls = 0
        self.total_weak_relationships = 0
        
        # Checkpoint files
        self.checkpoint_dir = f"checkpoints/{secret_word}"
        self.rankings_file = f"{self.checkpoint_dir}/rankings.json"
        self.clues_file = f"{self.checkpoint_dir}/clues.json"
        self.progress_file = f"{self.checkpoint_dir}/progress.json"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.words = []
        self.rankings = {}
        self.clues = {}
        self.progress = {
            'rankings_computed': False,
            'last_processed_rank': 0,
            'ai_cutoff_reached': False,
            'cutoff_rank': None,
            'total_ai_calls': 0,
            'consecutive_weak_count': 0
        }
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI()
            self.use_ai = True
            print("✅ OpenAI client initialized")
        except Exception as e:
            print(f"⚠️  OpenAI not available: {e}")
            self.use_ai = False
    
    def save_checkpoint(self):
        """Save current progress to checkpoint files"""
        print(f"💾 Saving checkpoint...")
        
        # Save progress
        self.progress.update({
            'ai_cutoff_reached': self.ai_cutoff_reached,
            'cutoff_rank': self.cutoff_rank,
            'total_ai_calls': self.total_ai_calls,
            'consecutive_weak_count': self.consecutive_weak_count,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)
        
        # Save clues
        with open(self.clues_file, 'w', encoding='utf-8') as f:
            json.dump(self.clues, f, indent=2)
        
        print(f"✅ Checkpoint saved to {self.checkpoint_dir}")
    
    def load_checkpoint(self):
        """Load progress from checkpoint files"""
        if os.path.exists(self.progress_file):
            print(f"📂 Loading checkpoint from {self.checkpoint_dir}")
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            
            # Load clues if they exist
            if os.path.exists(self.clues_file):
                with open(self.clues_file, 'r', encoding='utf-8') as f:
                    self.clues = json.load(f)
            
            # Restore state
            self.ai_cutoff_reached = self.progress.get('ai_cutoff_reached', False)
            self.cutoff_rank = self.progress.get('cutoff_rank')
            self.total_ai_calls = self.progress.get('total_ai_calls', 0)
            self.consecutive_weak_count = self.progress.get('consecutive_weak_count', 0)
            
            print(f"✅ Loaded checkpoint: {len(self.clues)} clues, rank {self.progress.get('last_processed_rank', 0)}")
            return True
        
        return False
    
    def load_words_and_rankings(self):
        """Load words and compute/load rankings"""
        # Load words from enable2.txt
        print("📚 Loading ENABLE2 word list...")
        try:
            with open("data/enable2.txt", 'r', encoding='utf-8') as f:
                self.words = [w.strip().lower() for w in f.readlines()]
            print(f"✅ Loaded {len(self.words):,} singular words")
        except FileNotFoundError:
            print("❌ ENABLE2 word list not found. Please run create_enable2.py first.")
            return False
        
        # Check if rankings are already computed
        if os.path.exists(self.rankings_file) and self.progress.get('rankings_computed', False):
            print("📂 Loading cached rankings...")
            with open(self.rankings_file, 'r', encoding='utf-8') as f:
                self.rankings = json.load(f)
            print(f"✅ Loaded rankings for {len(self.rankings):,} words")
        else:
            # Compute rankings
            if not self.compute_rankings():
                return False
        
        return True
    
    def compute_rankings(self):
        """Compute semantic rankings using cached embeddings"""
        print("🧮 Computing semantic rankings...")
        
        # Load embeddings
        try:
            with open(".env/embeddings2.json", 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
            print(f"✅ Loaded embeddings for {len(embeddings):,} words")
        except FileNotFoundError:
            print("❌ Embeddings2 file not found. Please run create_embeddings2.py first.")
            return False
        
        if self.secret_word not in embeddings:
            print(f"❌ '{self.secret_word}' not found in embeddings")
            return False
        
        # Get secret word embedding and normalize
        secret_embedding = np.array(embeddings[self.secret_word])
        secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
        
        # Compute similarities
        word_similarities = []
        for word in tqdm(self.words, desc="Computing similarities"):
            if word in embeddings:
                word_embedding = np.array(embeddings[word])
                word_embedding = word_embedding / np.linalg.norm(word_embedding)
                similarity = np.dot(word_embedding, secret_embedding)
                word_similarities.append((word, similarity))
            else:
                word_similarities.append((word, -1.0))
        
        # Sort and create rankings
        word_similarities.sort(key=lambda x: (-x[1], x[0]))
        
        self.rankings = {}
        for rank, (word, similarity) in enumerate(word_similarities, 1):
            self.rankings[word] = {'rank': rank, 'similarity': similarity}
        
        # Save rankings
        with open(self.rankings_file, 'w', encoding='utf-8') as f:
            json.dump(self.rankings, f, indent=2)
        
        self.progress['rankings_computed'] = True
        
        print(f"✅ Computed rankings for {len(self.rankings):,} words")
        print(f"🎯 '{self.secret_word}' has rank: {self.rankings[self.secret_word]['rank']}")
        
        return True
    
    def get_clue_and_strength(self, guess_word):
        """Get clue and relationship strength for a word pair"""
        
        if self.ai_cutoff_reached or not self.use_ai:
            return {'clue': None, 'strength': 'cutoff', 'ai_used': False}
        
        try:
            self.total_ai_calls += 1
            
            prompt = f"""You must analyze the word '{guess_word}' and its relationship to '{self.secret_word}'. 

Return JSON with TWO fields:
1. "clue": A 7-word-or-less description of how they relate (use 'that animal/creature/thing' instead of '{self.secret_word}')
2. "strength": Rate the relationship strength as "strong", "medium", or "weak"

Relationship strength guide:
- "strong": Direct interaction, clear connection (leash, collar, bone, walk)
- "medium": Indirect connection, some logical link (tree, park, house)  
- "weak": Very distant, forced connection, opposite concepts (calculator, mathematics, philosophy)

Focus on HOW they connect, not what '{guess_word}' is. Even if distant, find some relationship.

Word to analyze: {guess_word}

Example format:
{{"clue": "connects to that animal for control", "strength": "strong"}}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            clue_data = json.loads(response_content)
            
            if "clue" in clue_data and "strength" in clue_data:
                clue = clue_data["clue"]
                strength = clue_data["strength"]
                
                # Update weak relationship tracking
                if strength == "weak":
                    self.consecutive_weak_count += 1
                    self.total_weak_relationships += 1
                    
                    if self.consecutive_weak_count >= self.consecutive_weak_threshold:
                        self.ai_cutoff_reached = True
                        self.cutoff_rank = self.rankings[guess_word]['rank']
                        print(f"\n🛑 AI Cutoff Reached at rank {self.cutoff_rank}!")
                        print(f"💰 {self.consecutive_weak_count} consecutive weak relationships detected.")
                        
                else:
                    self.consecutive_weak_count = 0
                
                return {
                    'clue': clue,
                    'strength': strength,
                    'ai_used': True,
                    'cutoff_reached': self.ai_cutoff_reached
                }
            else:
                return {'clue': 'ERROR', 'strength': 'error', 'ai_used': True}
                
        except Exception as e:
            print(f"⚠️  AI call failed for '{guess_word}': {e}")
            return {'clue': 'ERROR', 'strength': 'error', 'ai_used': True}
    
    def generate_clues_with_checkpoints(self, checkpoint_interval=100):
        """Generate clues with periodic checkpointing"""
        print(f"🤖 Generating clues with checkpoints every {checkpoint_interval} words...")
        
        # Get words sorted by rank
        ranked_words = [(word, data['rank']) for word, data in self.rankings.items()]
        ranked_words.sort(key=lambda x: x[1])
        
        # Secret word clue
        if self.secret_word not in self.clues:
            self.clues[self.secret_word] = "This is the *."
        
        # Find starting point
        start_idx = self.progress.get('last_processed_rank', 0)
        if start_idx > 0:
            print(f"📍 Resuming from rank {start_idx + 1}")
        
        # Process words starting from checkpoint
        for i, (word, rank) in enumerate(tqdm(ranked_words[start_idx:], desc="Generating clues"), start_idx):
            if word == self.secret_word:
                continue
                
            if word not in self.clues:
                result = self.get_clue_and_strength(word)
                
                if result:
                    self.clues[word] = result['clue']
                    
                    if result.get('cutoff_reached', False):
                        print(f"🛑 Cutoff reached at rank {rank}")
                        break
                
                # Periodic checkpoint saving
                if (i + 1) % checkpoint_interval == 0:
                    self.progress['last_processed_rank'] = i + 1
                    self.save_checkpoint()
                    print(f"💾 Checkpoint saved at rank {rank}")
        
        # Final checkpoint
        self.progress['last_processed_rank'] = len(ranked_words)
        self.save_checkpoint()
        
        return True
    
    def generate_csv(self, output_filename):
        """Generate final CSV file"""
        print(f"📄 Generating CSV: {output_filename}")
        
        csv_data = []
        for word in self.words:
            rank_info = self.rankings.get(word, {'rank': len(self.words) + 1})
            rank = rank_info['rank']
            
            # Determine clue
            if rank == 1:
                clue = "This is the *."
            elif self.cutoff_rank and rank >= self.cutoff_rank:
                clue = None  # NULL for words beyond cutoff
            else:
                clue = self.clues.get(word, None)
            
            csv_data.append({
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': clue
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.sort_values(by='rank', inplace=True)
        
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df.to_csv(output_filename, index=False)
        
        # Report results
        file_size = os.path.getsize(output_filename)
        ai_clues = len([c for c in df['clue'] if c and c not in ['This is the *.', 'ERROR']])
        null_clues = len([c for c in df['clue'] if c is None or pd.isna(c)])
        
        print(f"✅ CSV created: {output_filename}")
        print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"📊 Total rows: {len(df):,}")
        print(f"📈 AI clues: {ai_clues:,}, NULL clues: {null_clues:,}")
        
        if self.cutoff_rank:
            saved_calls = len(self.words) - self.cutoff_rank
            savings_percent = (saved_calls / len(self.words)) * 100
            print(f"💰 Dynamic cutoff saved ~{saved_calls:,} API calls ({savings_percent:.1f}%)")
        
        return True

def generate_word_csv(word):
    """Generate CSV for a specific word"""
    print(f"\n=== Generating {word.upper()} CSV with Dynamic Cutoff ===")
    
    generator = CheckpointSemanticGenerator(word, consecutive_weak_threshold=5)
    
    # Load checkpoint if exists
    generator.load_checkpoint()
    
    # Load words and rankings
    if not generator.load_words_and_rankings():
        return False
    
    # Generate clues with checkpointing
    if not generator.generate_clues_with_checkpoints(checkpoint_interval=50):
        return False
    
    # Generate final CSV
    output_file = f"secretword/secretword-easy-animals-{word}.csv"
    if not generator.generate_csv(output_file):
        return False
    
    print(f"🎉 Successfully generated {word} CSV with dynamic cutoff!")
    return True

def main():
    """Generate both cat and dog CSVs"""
    print("=== Generating Cat and Dog CSVs with Dynamic Cutoff ===")
    
    words_to_process = ['cat', 'dog']
    
    for word in words_to_process:
        try:
            success = generate_word_csv(word)
            if success:
                print(f"✅ {word.upper()} completed successfully")
            else:
                print(f"❌ {word.upper()} failed")
        except KeyboardInterrupt:
            print(f"\n⏸️  Process interrupted for {word}. Progress saved to checkpoints/{word}/")
            print(f"💡 Run this script again to resume from where you left off.")
            break
        except Exception as e:
            print(f"❌ Unexpected error processing {word}: {e}")
    
    print("\n🏁 Processing complete!")

if __name__ == "__main__":
    main()
