#!/usr/bin/env python3
"""
Create EMBEDDINGS2.json - a filtered version of embeddings.json that only contains
entries for words that exist in ENABLE2.txt (singular words only)
"""

import json
import os
from datetime import datetime

def log(msg):
    """Print timestamped log message with immediate flush."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {msg}', flush=True)

def create_embeddings2():
    """Create EMBEDDINGS2.json with only ENABLE2 words"""
    
    log("=== Creating EMBEDDINGS2.json (ENABLE2 words only) ===")
    
    # Step 1: Load ENABLE2 word list
    enable2_path = "data/enable2.txt"
    log(f"Loading ENABLE2 word list from {enable2_path}...")
    
    try:
        with open(enable2_path, 'r', encoding='utf-8') as f:
            enable2_words = set(word.strip().lower() for word in f.readlines())
        log(f"✅ Loaded {len(enable2_words):,} words from ENABLE2")
    except FileNotFoundError:
        log(f"❌ ENABLE2 not found at {enable2_path}")
        log("Please run create_enable2.py first to create the ENABLE2 word list")
        return False
    except Exception as e:
        log(f"❌ Error loading ENABLE2: {e}")
        return False
    
    # Step 2: Load original embeddings
    embeddings_path = ".env/embeddings.json"
    log(f"Loading original embeddings from {embeddings_path}...")
    
    if not os.path.exists(embeddings_path):
        log(f"❌ Original embeddings not found at {embeddings_path}")
        return False
    
    try:
        # Check file size first
        file_size = os.path.getsize(embeddings_path)
        log(f"📏 Original embeddings file size: {file_size:,} bytes ({file_size/1024/1024/1024:.1f} GB)")
        log("⏳ Loading embeddings (this may take several minutes)...")
        
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            original_embeddings = json.load(f)
        
        log(f"✅ Loaded {len(original_embeddings):,} embeddings from original file")
        
    except Exception as e:
        log(f"❌ Error loading original embeddings: {e}")
        return False
    
    # Step 3: Filter embeddings to only include ENABLE2 words
    log("🔍 Filtering embeddings to match ENABLE2 words...")
    
    filtered_embeddings = {}
    found_count = 0
    missing_count = 0
    
    for word in enable2_words:
        if word in original_embeddings:
            filtered_embeddings[word] = original_embeddings[word]
            found_count += 1
        else:
            missing_count += 1
            if missing_count <= 10:  # Show first 10 missing words
                log(f"⚠️  Missing embedding for: {word}")
    
    log(f"✅ Filtered embeddings:")
    log(f"   Found: {found_count:,} words")
    log(f"   Missing: {missing_count:,} words")
    log(f"   Coverage: {found_count/len(enable2_words)*100:.1f}%")
    
    # Step 4: Save filtered embeddings
    output_path = ".env/embeddings2.json"
    log(f"💾 Saving filtered embeddings to {output_path}...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_embeddings, f, separators=(',', ':'))
        
        # Check output file size
        output_size = os.path.getsize(output_path)
        original_size = os.path.getsize(embeddings_path)
        reduction = (1 - output_size/original_size) * 100
        
        log(f"✅ Successfully created {output_path}")
        log(f"📏 Output file size: {output_size:,} bytes ({output_size/1024/1024:.1f} MB)")
        log(f"📊 Size reduction: {reduction:.1f}% smaller than original")
        log(f"🎯 Contains embeddings for {len(filtered_embeddings):,} words")
        
        # Show some statistics
        if missing_count > 0:
            log(f"\n📝 Note: {missing_count:,} words from ENABLE2 don't have embeddings")
            log(f"This is normal - some words may not have been in the original embedding cache")
        
        return True
        
    except Exception as e:
        log(f"❌ Error saving filtered embeddings: {e}")
        return False

def main():
    """Main function"""
    log("Starting EMBEDDINGS2 creation process...")
    
    success = create_embeddings2()
    
    if success:
        log("🎉 EMBEDDINGS2.json created successfully!")
        log("📝 You can now use embeddings2.json for faster processing with ENABLE2 words")
    else:
        log("❌ EMBEDDINGS2 creation failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
