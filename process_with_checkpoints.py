#!/usr/bin/env python3
"""
Process secret word CSVs using ordered embeddings checkpoints
Implements proper weak connection queue with saves every 200 records
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
from datetime import datetime
import sys

def load_ordered_embeddings(secret_word):
    """Load ordered embeddings from checkpoint file"""
    
    embeddings_file = f"secretword/embeddings-{secret_word}.txt"
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Ordered embeddings file not found: {embeddings_file}")
        print(f"💡 Run: python create_ordered_embeddings.py to create it")
        return None
    
    print(f"📂 Loading ordered embeddings from {embeddings_file}")
    
    ranked_words = []
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                rank = int(parts[0])
                word = parts[1]
                similarity = float(parts[2])
                ranked_words.append((rank, word, similarity))
    
    print(f"✅ Loaded {len(ranked_words):,} ranked words")
    return ranked_words

def process_secret_word_with_queue(secret_word, consecutive_weak_threshold=5, hard_cutoff_rank=10000, save_interval=200):
    """Process a secret word CSV with proper weak connection queue"""
    
    print(f"\n=== Processing {secret_word.upper()} with Weak Connection Queue ===")
    print(f"📊 Hard cutoff at rank {hard_cutoff_rank:,}")
    print(f"💾 Saving every {save_interval} words")
    print(f"🔄 Weak connections queued after {consecutive_weak_threshold} consecutive")
    
    # File paths
    csv_file = f"secretword/secretword-easy-animals-{secret_word}.csv"
    backup_file = f"secretword/secretword-easy-animals-{secret_word}_backup.csv"
    temp_file = f"secretword/secretword-easy-animals-{secret_word}_temp.csv"
    queue_file = f"secretword/secretword-easy-animals-{secret_word}_weak_queue.json"
    progress_file = f"secretword/secretword-easy-animals-{secret_word}_progress.json"
    
    # Load ordered embeddings (checkpoint)
    ranked_words = load_ordered_embeddings(secret_word)
    if not ranked_words:
        return False
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        use_ai = True
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ OpenAI not available: {e}")
        return False
    
    # Load or initialize progress
    progress = {
        'last_processed_rank': 0,
        'words_processed': 0,
        'ai_calls_made': 0,
        'consecutive_weak_count': 0,
        'ai_cutoff_reached': False,
        'cutoff_rank': None,
        'start_time': datetime.now().isoformat()
    }
    
    if os.path.exists(progress_file):
        print(f"📂 Loading previous progress...")
        with open(progress_file, 'r', encoding='utf-8') as f:
            saved_progress = json.load(f)
            progress.update(saved_progress)
        print(f"✅ Resuming from rank {progress['last_processed_rank']}")
    
    # Initialize tracking
    weak_queue = []
    final_csv_data = []
    consecutive_weak_count = progress['consecutive_weak_count']
    ai_cutoff_reached = progress['ai_cutoff_reached']
    cutoff_rank = progress['cutoff_rank']
    total_ai_calls = progress['ai_calls_made']
    
    def get_clue_and_strength(guess_word, current_rank):
        """Get clue and relationship strength"""
        nonlocal consecutive_weak_count, ai_cutoff_reached, cutoff_rank, total_ai_calls
        
        if ai_cutoff_reached or current_rank > hard_cutoff_rank:
            return {'clue': None, 'strength': 'cutoff', 'ai_used': False}
        
        try:
            total_ai_calls += 1
            
            prompt = f"""You must analyze the word '{guess_word}' and its relationship to '{secret_word}'. 

Return JSON with TWO fields:
1. "clue": A 7-word-or-less description of how they relate (use 'that animal/creature/thing' instead of '{secret_word}')
2. "strength": Rate the relationship strength as "strong", "medium", or "weak"

Relationship strength guide:
- "strong": Direct interaction, clear connection (leash, collar, bone, walk)
- "medium": Indirect connection, some logical link (tree, park, house)  
- "weak": Very distant, forced connection, opposite concepts (calculator, mathematics, philosophy)

Focus on HOW they connect, not what '{guess_word}' is.

Word to analyze: {guess_word}

Example format:
{{"clue": "connects to that animal for control", "strength": "strong"}}"""

            response = client.chat.completions.create(
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
                    consecutive_weak_count += 1
                    if consecutive_weak_count >= consecutive_weak_threshold:
                        ai_cutoff_reached = True
                        cutoff_rank = current_rank
                        print(f"\n🛑 AI Cutoff Reached at rank {cutoff_rank}!")
                        print(f"💰 {consecutive_weak_count} consecutive weak relationships detected.")
                else:
                    consecutive_weak_count = 0
                
                return {'clue': clue, 'strength': strength, 'ai_used': True}
            else:
                return {'clue': 'ERROR', 'strength': 'error', 'ai_used': True}
                
        except Exception as e:
            print(f"⚠️  AI call failed for '{guess_word}': {e}")
            return {'clue': 'ERROR', 'strength': 'error', 'ai_used': True}
    
    def save_progress_checkpoint():
        """Save current progress"""
        # Update progress
        progress.update({
            'last_processed_rank': current_rank,
            'words_processed': len(final_csv_data),
            'ai_calls_made': total_ai_calls,
            'consecutive_weak_count': consecutive_weak_count,
            'ai_cutoff_reached': ai_cutoff_reached,
            'cutoff_rank': cutoff_rank,
            'last_save': datetime.now().isoformat()
        })
        
        # Save progress
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        
        # Save weak queue
        queue_data = {
            'secret_word': secret_word,
            'weak_connections': weak_queue,
            'metadata': {
                'total_queued': len(weak_queue),
                'last_updated': datetime.now().isoformat()
            }
        }
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(queue_data, f, indent=2)
        
        # Save current CSV data
        if final_csv_data:
            temp_df = pd.DataFrame(final_csv_data)
            temp_df.to_csv(temp_file, index=False)
        
        print(f"💾 Progress saved at rank {current_rank} ({len(final_csv_data)} rows, {len(weak_queue)} queued)")
    
    # Process words from checkpoint
    print(f"🤖 Processing words with weak connection queue...")
    
    current_rank = 1
    start_index = progress['last_processed_rank']
    
    # Add secret word if starting fresh
    if start_index == 0:
        final_csv_data.append({
            'rank': 1,
            'secret_word': secret_word,
            'word': secret_word,
            'clue': 'This is the *.',
            'connection_strength': 'secret_word'
        })
        current_rank = 2
        start_index = 1
    
    # Process remaining words
    words_processed = 0
    
    for i in tqdm(range(start_index, len(ranked_words)), desc="Processing words", initial=start_index, total=len(ranked_words)):
        original_rank, word, similarity = ranked_words[i]
        
        if ai_cutoff_reached or current_rank > hard_cutoff_rank:
            # Add remaining words with NULL clues
            final_csv_data.append({
                'rank': current_rank,
                'secret_word': secret_word,
                'word': word,
                'clue': None,
                'connection_strength': 'cutoff' if ai_cutoff_reached else 'hard_cutoff'
            })
            current_rank += 1
            
        else:
            # Get AI assessment
            result = get_clue_and_strength(word, current_rank)
            words_processed += 1
            
            if result and result['strength'] == 'weak':
                # Queue weak connection
                weak_entry = {
                    'word': word,
                    'original_rank': original_rank,
                    'current_rank': current_rank,
                    'clue': result['clue'],
                    'strength': 'weak',
                    'similarity': similarity,
                    'queued_at': datetime.now().isoformat()
                }
                weak_queue.append(weak_entry)
                
                print(f"    Rank {current_rank:>5}: {word:<15} → 🗂️  QUEUED (weak #{len(weak_queue)})")
                # Don't add to final_csv_data, don't increment current_rank
                
            else:
                # Keep this word
                final_csv_data.append({
                    'rank': current_rank,
                    'secret_word': secret_word,
                    'word': word,
                    'clue': result['clue'] if result['ai_used'] else None,
                    'connection_strength': result.get('strength', 'unknown')
                })
                
                if result.get('strength') in ['strong', 'medium']:
                    print(f"    Rank {current_rank:>5}: {word:<15} → ✅ {result.get('strength', 'unknown')}")
                
                current_rank += 1
        
        # Periodic save
        if (words_processed > 0 and words_processed % save_interval == 0) or ai_cutoff_reached:
            save_progress_checkpoint()
            
            if ai_cutoff_reached:
                print(f"🛑 Stopping processing due to AI cutoff")
                break
    
    # Insert weak connections at appropriate rank
    if weak_queue:
        insertion_rank = min(hard_cutoff_rank, current_rank)
        print(f"\n📥 Inserting {len(weak_queue)} weak connections at rank {insertion_rank}")
        
        # Adjust existing ranks
        for entry in final_csv_data:
            if entry['rank'] >= insertion_rank:
                entry['rank'] += len(weak_queue)
        
        # Add weak connections
        for i, weak_entry in enumerate(weak_queue):
            final_csv_data.append({
                'rank': insertion_rank + i,
                'secret_word': secret_word,
                'word': weak_entry['word'],
                'clue': 'weak connection',
                'connection_strength': 'weak'
            })
        
        print(f"✅ Inserted weak connections at ranks {insertion_rank}-{insertion_rank + len(weak_queue) - 1}")
    
    # Create final CSV
    print(f"💾 Creating final CSV...")
    final_df = pd.DataFrame(final_csv_data)
    final_df = final_df.sort_values('rank').reset_index(drop=True)
    final_df.to_csv(csv_file, index=False)
    
    # Clean up temp files
    for temp in [temp_file, progress_file]:
        if os.path.exists(temp):
            os.remove(temp)
    
    # Report results
    file_size = os.path.getsize(csv_file)
    ai_clues = len([c for c in final_df['clue'] if c and c not in ['This is the *.', 'ERROR', 'weak connection'] and pd.notna(c)])
    weak_connection_clues = len([c for c in final_df['clue'] if c == 'weak connection'])
    null_clues = len([c for c in final_df['clue'] if c is None or pd.isna(c)])
    
    print(f"\n📊 Processing Results for {secret_word.upper()}:")
    print(f"   Total words processed: {words_processed:,}")
    print(f"   Total AI calls: {total_ai_calls:,}")
    print(f"   Final CSV rows: {len(final_df):,}")
    print(f"   AI clues: {ai_clues:,}")
    print(f"   Weak connections: {weak_connection_clues:,}")
    print(f"   NULL clues: {null_clues:,}")
    print(f"   File size: {file_size/1024/1024:.1f} MB")
    
    if cutoff_rank:
        print(f"💰 Dynamic cutoff at rank {cutoff_rank:,}")
    
    print(f"✅ {secret_word.upper()} processing complete!")
    return True

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python process_with_checkpoints.py <secret_word>")
        print("Example: python process_with_checkpoints.py cat")
        return
    
    secret_word = sys.argv[1].lower()
    
    try:
        success = process_secret_word_with_queue(secret_word)
        if success:
            print(f"\n🎉 {secret_word.upper()} processed successfully!")
        else:
            print(f"\n❌ {secret_word.upper()} processing failed!")
    except KeyboardInterrupt:
        print(f"\n⏸️  Processing interrupted. Progress saved for {secret_word}")
    except Exception as e:
        print(f"\n❌ Error processing {secret_word}: {e}")

if __name__ == "__main__":
    main()
