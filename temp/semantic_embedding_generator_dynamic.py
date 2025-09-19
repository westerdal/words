#!/usr/bin/env python3
"""
Dynamic Semantic Rank Generator - Uses OpenAI embeddings with dynamic AI cutoff
Integrates relationship strength detection to optimize API usage
"""

import os
import json
import time
import math
import pandas as pd
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DynamicSemanticEmbeddingGenerator:
    def __init__(self, secret_word, consecutive_weak_threshold=5, openai_api_key=None):
        self.secret_word = secret_word.lower()
        self.consecutive_weak_threshold = consecutive_weak_threshold
        self.consecutive_weak_count = 0
        self.ai_cutoff_reached = False
        self.cutoff_rank = None
        self.total_ai_calls = 0
        self.total_weak_relationships = 0
        
        self.words = []
        self.rankings = {}
        self.client = None
        self.use_ai = False
        
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
            self.use_ai = True
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_ai = True
        else:
            print("⚠️  Warning: No OPENAI_API_KEY found in environment variables or provided.")
            print("Will generate CSV without AI clues (NULL values only).")

    def load_words(self, filename="data/enable2.txt"):
        """Load words from the ENABLE word list (singular words only)."""
        print(f"📚 Loading words from {filename}...")
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.words = [word.strip().lower() for word in file.readlines()]
            print(f"✅ Loaded {len(self.words):,} singular words")
            
            if self.secret_word not in self.words:
                print(f"❌ Error: Secret word '{self.secret_word}' not found in word list!")
                return False
            return True
        except FileNotFoundError:
            print(f"❌ Error: Could not find word list file at {filename}")
            print("💡 Please run create_enable2.py to generate the singular word list")
            self.words = []
            return False

    def load_embeddings_cache(self, cache_path=".env/embeddings2.json"):
        """Load precomputed embeddings from cache file."""
        print(f"🧮 Loading embeddings cache from {cache_path}...")
        
        try:
            if not os.path.exists(cache_path):
                print(f"❌ Cache file not found at {cache_path}")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            print(f"✅ Loaded embeddings for {len(cache):,} words from cache")
            return cache
            
        except Exception as e:
            print(f"❌ Error loading embeddings cache: {e}")
            return None

    def compute_semantic_rankings_from_cache(self, cache_path=".env/embeddings2.json"):
        """Compute semantic rankings using cached embeddings."""
        print("🔍 Computing semantic rankings from cached embeddings...")
        
        # Load cached embeddings
        embeddings_cache = self.load_embeddings_cache(cache_path)
        if embeddings_cache is None:
            print("❌ Failed to load embeddings cache")
            return False
        
        # Check if secret word is in cache
        if self.secret_word not in embeddings_cache:
            print(f"❌ Secret word '{self.secret_word}' not found in embeddings cache")
            return False
        
        # Get secret word embedding and normalize
        secret_embedding = np.array(embeddings_cache[self.secret_word])
        secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
        
        # Compute similarities for all words
        word_similarities = []
        missing_embeddings = 0
        
        print(f"🔄 Computing similarities for {len(self.words):,} words...")
        for word in tqdm(self.words, desc="Computing similarities"):
            if word in embeddings_cache:
                # Get cached embedding and normalize
                word_embedding = np.array(embeddings_cache[word])
                word_embedding = word_embedding / np.linalg.norm(word_embedding)
                
                # Compute cosine similarity
                similarity = np.dot(word_embedding, secret_embedding)
                word_similarities.append((word, similarity))
            else:
                # Word not in cache - assign very low similarity
                word_similarities.append((word, -1.0))
                missing_embeddings += 1
        
        if missing_embeddings > 0:
            print(f"⚠️  {missing_embeddings:,} words missing from embeddings cache")
        
        # Sort by similarity (descending), then alphabetically for ties
        word_similarities.sort(key=lambda x: (-x[1], x[0]))
        
        # Create rankings dictionary
        self.rankings = {}
        for rank, (word, similarity) in enumerate(word_similarities, 1):
            self.rankings[word] = {'rank': rank, 'similarity': similarity}
        
        print(f"✅ Computed rankings for {len(self.rankings):,} words")
        print(f"🎯 '{self.secret_word}' has rank: {self.rankings[self.secret_word]['rank']}")
        
        # Show top 10 for verification
        top_10 = word_similarities[:10]
        print(f"\nTop 10 most similar words to '{self.secret_word}':")
        for i, (word, sim) in enumerate(top_10, 1):
            print(f"  {i:2d}. {word:<15} (similarity: {sim:.6f})")
        
        return True

    def get_clue_and_strength(self, guess_word):
        """
        Get clue and relationship strength for a word pair
        
        Returns:
            dict with 'clue', 'strength', 'ai_used' fields, or None if AI cutoff reached
        """
        
        # Check if AI cutoff has been reached
        if self.ai_cutoff_reached:
            return {
                'clue': None,  # NULL clue
                'strength': 'cutoff',
                'ai_used': False,
                'reason': f'AI cutoff reached after {self.consecutive_weak_count} consecutive weak relationships'
            }
        
        # Check if AI is available
        if not self.use_ai:
            return {
                'clue': None,  # NULL clue
                'strength': 'no_ai',
                'ai_used': False,
                'reason': 'OpenAI not available'
            }
        
        # Make AI call
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
                    
                    # Check if we've reached the cutoff threshold
                    if self.consecutive_weak_count >= self.consecutive_weak_threshold:
                        self.ai_cutoff_reached = True
                        print(f"\n🛑 AI Cutoff Reached! {self.consecutive_weak_count} consecutive weak relationships detected.")
                        print(f"💰 Future API calls will be skipped. Total AI calls made: {self.total_ai_calls}")
                        
                else:
                    # Reset consecutive count if we get a non-weak relationship
                    self.consecutive_weak_count = 0
                
                return {
                    'clue': clue,
                    'strength': strength,
                    'ai_used': True,
                    'consecutive_weak': self.consecutive_weak_count,
                    'cutoff_reached': self.ai_cutoff_reached
                }
            else:
                # Fallback for malformed response
                return {
                    'clue': 'ERROR',
                    'strength': 'error',
                    'ai_used': True,
                    'reason': 'Malformed AI response'
                }
                
        except Exception as e:
            print(f"⚠️  AI call failed for '{guess_word}': {e}")
            return {
                'clue': 'ERROR',
                'strength': 'error',
                'ai_used': True,
                'reason': f'API error: {e}'
            }

    def generate_csv_with_dynamic_cutoff(self, output_filename):
        """Generate CSV file with dynamic AI cutoff system."""
        print(f"📄 Generating CSV with dynamic cutoff: {output_filename}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Get words sorted by rank for processing
        ranked_words = [(word, data['rank']) for word, data in self.rankings.items()]
        ranked_words.sort(key=lambda x: x[1])  # Sort by rank
        
        # Generate clues with dynamic cutoff
        clues = {}
        ai_clue_count = 0
        
        # Special case for secret word (rank 1)
        clues[self.secret_word] = "This is the *."
        
        print(f"🤖 Generating AI clues with dynamic cutoff...")
        
        for word, rank in tqdm(ranked_words[1:], desc="Generating clues"):  # Skip secret word
            result = self.get_clue_and_strength(word)
            
            if result:
                if result['ai_used'] and not result.get('cutoff_reached', False):
                    # AI clue generated successfully
                    clues[word] = result['clue']
                    ai_clue_count += 1
                    
                elif result.get('cutoff_reached', False):
                    # Cutoff reached - record the rank and stop
                    self.cutoff_rank = rank
                    print(f"🛑 Dynamic cutoff triggered at rank {rank} (word: '{word}')")
                    break
                else:
                    # Error or no AI available
                    clues[word] = result['clue']
            else:
                # No result - use NULL
                clues[word] = None
        
        # Create CSV data
        csv_data = []
        for word in self.words:
            rank_info = self.rankings.get(word, {'rank': len(self.words) + 1})
            rank = rank_info['rank']
            
            # Determine clue based on rank and cutoff
            if rank == 1:
                clue = "This is the *."
            elif self.cutoff_rank and rank >= self.cutoff_rank:
                clue = None  # NULL for words beyond cutoff
            else:
                clue = clues.get(word, None)
            
            csv_data.append({
                'rank': rank,
                'secret_word': self.secret_word,
                'word': word,
                'clue': clue
            })
        
        # Create DataFrame and sort by rank
        df = pd.DataFrame(csv_data)
        df.sort_values(by='rank', inplace=True)
        
        # Save CSV
        df.to_csv(output_filename, index=False)
        
        # Report results
        file_size = os.path.getsize(output_filename)
        print(f"✅ CSV created: {output_filename}")
        print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"📊 Total rows: {len(df):,}")
        
        # Count clue types
        ai_clues = len([c for c in df['clue'] if c and c not in ['This is the *.', 'ERROR']])
        error_clues = len([c for c in df['clue'] if c == 'ERROR'])
        null_clues = len([c for c in df['clue'] if c is None or pd.isna(c)])
        
        print(f"📈 Clue breakdown:")
        print(f"   AI-generated: {ai_clues:,}")
        print(f"   Error clues: {error_clues:,}")
        print(f"   NULL clues: {null_clues:,}")
        
        # Dynamic cutoff statistics
        print(f"\n📊 Dynamic Cutoff Statistics:")
        print(f"   Total AI calls made: {self.total_ai_calls:,}")
        print(f"   Weak relationships found: {self.total_weak_relationships}")
        print(f"   Cutoff triggered: {'Yes' if self.ai_cutoff_reached else 'No'}")
        if self.cutoff_rank:
            saved_calls = len(self.words) - self.cutoff_rank
            savings_percent = (saved_calls / len(self.words)) * 100
            print(f"   Cutoff rank: {self.cutoff_rank:,}")
            print(f"   API calls saved: ~{saved_calls:,} ({savings_percent:.1f}%)")
            print(f"💡 Dynamic cutoff saved significant API costs while maintaining quality!")
        
        return True

    def get_stats(self):
        """Get statistics about AI usage and cutoff performance"""
        return {
            'total_ai_calls': self.total_ai_calls,
            'total_weak_relationships': self.total_weak_relationships,
            'consecutive_weak_count': self.consecutive_weak_count,
            'ai_cutoff_reached': self.ai_cutoff_reached,
            'consecutive_weak_threshold': self.consecutive_weak_threshold,
            'cutoff_rank': self.cutoff_rank,
            'total_words': len(self.words)
        }

def main():
    """Main function for testing"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python semantic_embedding_generator_dynamic.py <secret_word>")
        return False
    
    secret_word = sys.argv[1].lower()
    print(f"=== Dynamic Semantic Embedding Generator ===")
    print(f"Secret word: {secret_word}")
    
    # Create generator
    generator = DynamicSemanticEmbeddingGenerator(secret_word, consecutive_weak_threshold=5)
    
    # Load words
    if not generator.load_words("data/enable2.txt"):
        return False
    
    # Compute rankings
    if not generator.compute_semantic_rankings_from_cache(".env/embeddings2.json"):
        print("❌ Failed to compute semantic rankings")
        return False
    
    # Generate CSV
    output_file = f"secretword/secretword-easy-test-{secret_word}.csv"
    if generator.generate_csv_with_dynamic_cutoff(output_file):
        print(f"\n🎉 Successfully generated {output_file}")
        return True
    else:
        print(f"\n❌ Failed to generate CSV")
        return False

if __name__ == "__main__":
    main()
