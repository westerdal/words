#!/usr/bin/env python3
"""
Process semantic rank words with dynamic AI cutoff system
Integrates relationship strength detection to optimize API usage
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import math
from tqdm import tqdm

class DynamicAICutoffSystem:
    """System that dynamically determines when to stop making AI calls based on relationship strength"""
    
    def __init__(self, consecutive_weak_threshold=5):
        """
        Initialize the cutoff system
        
        Args:
            consecutive_weak_threshold: Number of consecutive weak relationships before stopping AI calls
        """
        self.consecutive_weak_threshold = consecutive_weak_threshold
        self.consecutive_weak_count = 0
        self.ai_cutoff_reached = False
        self.total_ai_calls = 0
        self.total_weak_relationships = 0
        self.cutoff_rank = None
        
        try:
            self.client = OpenAI()
            self.ai_available = True
            print("✅ OpenAI client initialized")
        except Exception as e:
            print(f"⚠️  OpenAI not available: {e}")
            self.ai_available = False
    
    def get_clue_and_strength(self, guess_word, secret_word):
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
        if not self.ai_available:
            return {
                'clue': 'ERROR',
                'strength': 'no_ai',
                'ai_used': False,
                'reason': 'OpenAI not available'
            }
        
        # Make AI call
        try:
            self.total_ai_calls += 1
            
            prompt = f"""You must analyze the word '{guess_word}' and its relationship to '{secret_word}'. 

Return JSON with TWO fields:
1. "clue": A 7-word-or-less description of how they relate (use 'that animal/creature/thing' instead of '{secret_word}')
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
                        self.cutoff_rank = "determined_dynamically"  # Will be set by caller
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
    
    def get_stats(self):
        """Get statistics about AI usage and cutoff performance"""
        return {
            'total_ai_calls': self.total_ai_calls,
            'total_weak_relationships': self.total_weak_relationships,
            'consecutive_weak_count': self.consecutive_weak_count,
            'ai_cutoff_reached': self.ai_cutoff_reached,
            'consecutive_weak_threshold': self.consecutive_weak_threshold,
            'cutoff_rank': self.cutoff_rank
        }

def is_plural(word):
    """
    Determines if a given word is likely plural based on a comprehensive
    set of English pluralization rules.

    Args:
      word: A string representing the word to check.

    Returns:
      True if the word is likely plural, False otherwise.
    """
    # Convert word to lowercase for case-insensitive checks
    w_lower = word.lower()

    # Rule 5: Check against a set of common irregular plurals
    irregular_plurals = {"men", "women", "children", "feet", "teeth", "mice", "people"}
    if w_lower in irregular_plurals:
        return True

    # Rule 6: Check against nouns where singular and plural forms are the same
    no_change_nouns = {"sheep", "deer", "fish", "series", "species"}
    if w_lower in no_change_nouns:
        return True

    # Rule 4: Check for words ending in "-ves" (e.g., wives, leaves)
    if w_lower.endswith("ves"):
        return True
        
    # Rule 3: Check for words ending in "-ies" (e.g., babies, parties)
    if w_lower.endswith("ies"):
        return True
        
    # Rule 2: Check for words ending in "-es" (e.g., boxes, buses)
    if w_lower.endswith("es"):
        return True
        
    # Rule 1: Check for the most common "-s" ending, excluding common singulars
    # to reduce false positives (e.g., "status", "bus", "class").
    if w_lower.endswith("s") and not w_lower.endswith(("ss", "us", "is")):
        return True

    # If none of the plural rules match, assume the word is singular.
    return False

def process_first_unprocessed_word():
    """Process the first unprocessed word from master-list.txt"""
    
    print("=== Dynamic Semantic Rank Word Processing ===")
    
    # Step 1: Read master-list.txt - Get first unprocessed word
    master_list_path = "secretword/master-list.txt"
    
    try:
        with open(master_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("❌ Master list is empty")
            return False
            
        print(f"📋 Found {len(lines)} words in master list")
        
    except FileNotFoundError:
        print(f"❌ Master list not found: {master_list_path}")
        return False
    
    # Find first unprocessed word
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Step 2: Parse format - Extract difficulty, category, word
        try:
            parts = line.split('-')
            if len(parts) != 3:
                print(f"⚠️  Skipping invalid format: {line}")
                continue
                
            difficulty, category, word = parts
            print(f"📝 Processing word {i+1}: {word} ({difficulty}, {category})")
            
        except Exception as e:
            print(f"⚠️  Error parsing line '{line}': {e}")
            continue
        
        # Step 3: Check if exists - Skip if CSV already exists
        csv_file = f"secretword/secretword-{difficulty}-{category}-{word}.csv"
        if os.path.exists(csv_file):
            print(f"✅ Already exists: {csv_file} - skipping")
            continue
        
        # Found first unprocessed word - process it
        print(f"🎯 Processing: {word}")
        success = process_single_word_dynamic(difficulty, category, word)
        
        if success:
            print(f"✅ Successfully processed '{word}'")
            return True
        else:
            print(f"❌ Failed to process '{word}'")
            return False
    
    print("✅ All words in master list have been processed")
    return True

def process_single_word_dynamic(difficulty, category, word):
    """Process a single word to create its CSV file with dynamic AI cutoff"""
    
    print(f"\n--- Processing {word} with Dynamic AI Cutoff ---")
    
    # Step 4: Load word list from data/enable2.txt (singular words only)
    print("📚 Loading ENABLE2 word list (singular words only)...")
    try:
        with open("data/enable2.txt", 'r', encoding='utf-8') as f:
            enable_words = [w.strip().lower() for w in f.readlines()]
        print(f"✅ Loaded {len(enable_words):,} singular words from ENABLE2 list")
        
        if word not in enable_words:
            print(f"❌ Error: Secret word '{word}' not found in ENABLE2 word list")
            return False
            
    except FileNotFoundError:
        print("❌ ENABLE2 word list not found at data/enable2.txt")
        print("💡 Please run create_enable2.py to generate the singular word list")
        return False
    
    # Step 5: Compute rankings - Use .env/embeddings2.json
    print("🧮 Loading embeddings and computing semantic similarity rankings...")
    
    try:
        # Load embeddings
        with open(".env/embeddings2.json", 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        print(f"✅ Loaded embeddings for {len(embeddings):,} words")
        
    except FileNotFoundError:
        print("❌ Embeddings file not found at .env/embeddings2.json")
        print("💡 Please run create_embeddings2.py to generate the filtered embeddings")
        return False
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Check if secret word is in embeddings
    if word not in embeddings:
        print(f"❌ '{word}' not found in embeddings")
        return False
    
    # Get secret word embedding and normalize
    secret_embedding = np.array(embeddings[word])
    secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
    
    # Compute similarities for all ENABLE words
    print(f"🔍 Computing similarities for {len(enable_words):,} words...")
    word_similarities = []
    
    for enable_word in tqdm(enable_words, desc="Computing similarities"):
        if enable_word in embeddings:
            # Get cached embedding
            word_embedding = np.array(embeddings[enable_word])
            word_embedding = word_embedding / np.linalg.norm(word_embedding)
            
            # Compute cosine similarity
            similarity = np.dot(word_embedding, secret_embedding)
            word_similarities.append((enable_word, similarity))
        else:
            # Word not in cache - assign very low similarity
            word_similarities.append((enable_word, -1.0))
    
    # Sort by similarity (descending), then alphabetically for ties
    word_similarities.sort(key=lambda x: (-x[1], x[0]))
    
    # Create rankings dictionary
    rankings = {}
    for rank, (ranked_word, similarity) in enumerate(word_similarities, 1):
        rankings[ranked_word] = {'rank': rank, 'similarity': similarity}
    
    print(f"✅ Computed rankings for {len(rankings):,} words")
    print(f"🎯 '{word}' has rank: {rankings[word]['rank']}")
    
    # Show top 10 for verification
    top_10 = word_similarities[:10]
    print(f"\nTop 10 most similar words to '{word}':")
    for i, (w, sim) in enumerate(top_10, 1):
        print(f"  {i:2d}. {w:<15} (similarity: {sim:.6f})")
    
    # Step 6: Generate clues with dynamic cutoff system
    print(f"\n🤖 Generating AI clues with dynamic cutoff system...")
    
    # Initialize dynamic cutoff system
    cutoff_system = DynamicAICutoffSystem(consecutive_weak_threshold=5)
    
    # Get words sorted by rank for processing
    ranked_words = [(w, data['rank']) for w, data in rankings.items()]
    ranked_words.sort(key=lambda x: x[1])  # Sort by rank
    
    # Generate clues with dynamic cutoff
    clues = {}
    ai_clue_count = 0
    cutoff_rank = None
    
    # Special case for secret word (rank 1)
    clues[word] = "This is the *."
    
    print(f"🔄 Processing words in rank order...")
    
    for ranked_word, rank in tqdm(ranked_words[1:], desc="Generating clues"):  # Skip secret word
        result = cutoff_system.get_clue_and_strength(ranked_word, word)
        
        if result:
            if result['ai_used'] and not result.get('cutoff_reached', False):
                # AI clue generated successfully
                clues[ranked_word] = result['clue']
                ai_clue_count += 1
                
                # Show progress for interesting relationships
                if result['strength'] in ['strong', 'medium'] or result.get('consecutive_weak', 0) > 0:
                    strength_emoji = {'strong': '💪', 'medium': '📊', 'weak': '⚠️ '}[result['strength']]
                    consecutive = result.get('consecutive_weak', 0)
                    if consecutive > 0:
                        print(f"    Rank {rank:>5}: {ranked_word:<12} → {strength_emoji} {result['strength']} (consecutive weak: {consecutive})")
                
            elif result.get('cutoff_reached', False):
                # Cutoff reached - record the rank and stop
                cutoff_rank = rank
                cutoff_system.cutoff_rank = rank
                print(f"🛑 Dynamic cutoff triggered at rank {rank} (word: '{ranked_word}')")
                break
            else:
                # Error or no AI available
                clues[ranked_word] = result['clue']
        else:
            # No result - use error
            clues[ranked_word] = 'ERROR'
    
    # Get final statistics
    stats = cutoff_system.get_stats()
    
    print(f"\n📊 Dynamic Cutoff Statistics:")
    print(f"   Total AI calls made: {stats['total_ai_calls']:,}")
    print(f"   AI clues generated: {ai_clue_count:,}")
    print(f"   Weak relationships found: {stats['total_weak_relationships']}")
    print(f"   Cutoff triggered: {'Yes' if stats['ai_cutoff_reached'] else 'No'}")
    if cutoff_rank:
        print(f"   Cutoff rank: {cutoff_rank:,}")
        total_words = len(enable_words)
        saved_calls = total_words - cutoff_rank
        savings_percent = (saved_calls / total_words) * 100
        print(f"   API calls saved: ~{saved_calls:,} ({savings_percent:.1f}%)")
    
    # Step 7: Create CSV - Save with proper filename and format
    print(f"\n📄 Creating CSV file...")
    
    csv_data = []
    for enable_word in enable_words:
        rank_info = rankings.get(enable_word, {'rank': len(enable_words) + 1})
        rank = rank_info['rank']
        
        # Determine clue based on rank and cutoff
        if rank == 1:
            clue = "This is the *."
        elif cutoff_rank and rank >= cutoff_rank:
            clue = None  # NULL for words beyond cutoff
        else:
            clue = clues.get(enable_word, 'ERROR')
        
        csv_data.append({
            'rank': rank,
            'secret_word': word,
            'word': enable_word,
            'clue': clue
        })
    
    # Create DataFrame and sort by rank
    df = pd.DataFrame(csv_data)
    df.sort_values(by='rank', inplace=True)
    
    # Save CSV
    csv_filename = f"secretword/secretword-{difficulty}-{category}-{word}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    
    # Report results
    file_size = os.path.getsize(csv_filename)
    print(f"✅ CSV created: {csv_filename}")
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
    
    # Show sample rows
    print(f"\nSample CSV content:")
    print(df.head(10).to_string(index=False))
    
    if cutoff_rank:
        print(f"\n💡 Dynamic cutoff saved approximately {len(enable_words) - cutoff_rank:,} API calls!")
        print(f"   This represents significant cost savings while maintaining quality for meaningful relationships.")
    
    return True

def main():
    """Main function"""
    print("Starting Dynamic Semantic Rank word processing...")
    success = process_first_unprocessed_word()
    
    if success:
        print("\n🎉 Processing completed successfully!")
    else:
        print("\n❌ Processing failed!")
    
    return success

if __name__ == "__main__":
    main()
