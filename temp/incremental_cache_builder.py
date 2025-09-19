#!/usr/bin/env python3
"""
Incremental Embedding Cache Builder - Saves each word individually
"""

import os
import json
import time
import sys
from datetime import datetime
from openai import OpenAI

def print_status(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {msg}')
    sys.stdout.flush()

def load_words(filename="data/enable1.txt"):
    with open(filename, 'r', encoding='utf-8') as file:
        return [word.strip().lower() for word in file.readlines()]

def get_cached_words():
    """Get list of words that are already cached."""
    cache_dir = "embedding_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        return set()
    
    cached_words = set()
    for filename in os.listdir(cache_dir):
        if filename.endswith('.json'):
            word = filename[:-5]  # Remove .json extension
            cached_words.add(word)
    
    return cached_words

def save_word_embedding(word, embedding):
    """Save individual word embedding to separate file."""
    cache_dir = "embedding_cache"
    filepath = os.path.join(cache_dir, f"{word}.json")
    with open(filepath, 'w') as f:
        json.dump(embedding, f, separators=(',', ':'))

def load_all_embeddings():
    """Load all cached embeddings into memory."""
    cache_dir = "embedding_cache"
    if not os.path.exists(cache_dir):
        return {}
    
    cache = {}
    for filename in os.listdir(cache_dir):
        if filename.endswith('.json'):
            word = filename[:-5]
            filepath = os.path.join(cache_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    cache[word] = json.load(f)
            except:
                continue
    return cache

def main():
    print_status('🚀 INCREMENTAL embedding cache builder')
    print_status('💾 Saves each word individually - no corruption risk!')
    
    # Initialize
    print_status('🔌 Creating OpenAI client...')
    client = OpenAI()
    print_status('✅ Client ready')
    
    # Load words and check cache
    print_status('📖 Loading words...')
    all_words = load_words()
    print_status(f'✅ Loaded {len(all_words):,} words')
    
    print_status('📂 Checking cached words...')
    cached_words = get_cached_words()
    print_status(f'✅ Found {len(cached_words):,} cached words')
    
    # Calculate remaining work
    remaining_words = [w for w in all_words if w not in cached_words]
    print_status(f'📊 Need embeddings for {len(remaining_words):,} words')
    
    if not remaining_words:
        print_status('🎉 All words already cached!')
        return
    
    # Processing setup
    batch_size = 20
    total_batches = (len(remaining_words) + batch_size - 1) // batch_size
    
    print_status(f'📊 PROCESSING PLAN:')
    print_status(f'   Words remaining: {len(remaining_words):,}')
    print_status(f'   Batches: {total_batches:,} x {batch_size} words')
    print_status(f'   Individual file saves: INSTANT (no hang risk)')
    
    # Start processing
    start_time = time.time()
    last_status_time = start_time
    successful = 0
    failed = 0
    
    print_status('🚀 STARTING INCREMENTAL PROCESSING...')
    
    for i in range(0, len(remaining_words), batch_size):
        batch_num = (i // batch_size) + 1
        batch_words = remaining_words[i:i + batch_size]
        current_time = time.time()
        
        # Status update every 30 seconds OR every 20 batches
        if (current_time - last_status_time) >= 30 or batch_num % 20 == 0:
            elapsed_min = (current_time - start_time) / 60
            progress_pct = (batch_num / total_batches) * 100
            words_per_min = successful / elapsed_min if elapsed_min > 0 else 0
            
            print_status('📊 === STATUS UPDATE ===')
            print_status(f'   Progress: {progress_pct:.1f}% (batch {batch_num:,}/{total_batches:,})')
            print_status(f'   Cached: {successful:,} words successfully')
            print_status(f'   Failed: {failed:,} words')
            print_status(f'   Rate: {words_per_min:.0f} words/minute')
            print_status(f'   Elapsed: {elapsed_min:.1f} minutes')
            
            if words_per_min > 0:
                remaining_count = len(remaining_words) - successful - failed
                eta_min = remaining_count / words_per_min
                print_status(f'   ETA: {eta_min:.0f} minutes')
            print_status('========================')
            last_status_time = current_time
        
        # Process batch
        try:
            print_status(f'🔄 Processing batch {batch_num}: {batch_words}')
            
            response = client.embeddings.create(
                model='text-embedding-3-large',
                input=batch_words,
                encoding_format='float'
            )
            embeddings = [data.embedding for data in response.data]
            
            print_status('💾 Saving individual embeddings...')
            # Save each word individually (INSTANT - no hang risk)
            for word, emb in zip(batch_words, embeddings):
                save_word_embedding(word, emb)
            
            successful += len(batch_words)
            print_status(f'✅ Batch {batch_num} completed successfully')
            
        except Exception as e:
            print_status(f'❌ Batch {batch_num} failed: {e}')
            failed += len(batch_words)
            print_status('⏳ Waiting 10 seconds before retry...')
            time.sleep(10)
        
        # Brief pause
        time.sleep(0.5)
    
    elapsed_min = (time.time() - start_time) / 60
    total_cached = len(get_cached_words())
    
    print_status('🎉 === FINAL RESULTS ===')
    print_status(f'⏱️  Total time: {elapsed_min:.1f} minutes')
    print_status(f'✅ Successfully cached: {successful:,} words this session')
    print_status(f'📁 Total cached words: {total_cached:,}')
    print_status(f'❌ Failed: {failed:,} words')
    print_status('🚀 INCREMENTAL CACHE BUILD COMPLETE!')
    
    # Optionally create consolidated cache file
    if total_cached > 1000:
        print_status('📦 Creating consolidated cache file...')
        all_cache = load_all_embeddings()
        with open('embeddings_cache.json', 'w') as f:
            json.dump(all_cache, f, separators=(',', ':'))
        print_status(f'✅ Consolidated cache saved: {len(all_cache):,} embeddings')

if __name__ == "__main__":
    main()

