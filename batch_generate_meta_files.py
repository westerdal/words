#!/usr/bin/env python3
"""
Batch generate meta files for all words that have embeddings but no meta files yet
"""

import os
import sys
import subprocess
from pathlib import Path
import re
import time

def scan_words_needing_meta():
    """Find words that have embeddings but no meta files"""
    secretword_dir = Path("secretword")
    
    # Get all embedding files and extract base words
    embedding_files = []
    for file in secretword_dir.glob("*-embeddings.txt"):
        word = file.stem.replace("-embeddings", "")
        # Remove suffixes like '2', '-clean' to get base word
        base_word = re.sub(r'-clean$', '', word)
        base_word = re.sub(r'2$', '', base_word)
        # Only include if it's a valid base word (no numbers or special chars)
        if base_word.isalpha() and len(base_word) > 1:
            embedding_files.append(base_word)
    
    embedding_words = sorted(set(embedding_files))
    print(f"📂 Found {len(embedding_words)} words with embeddings")
    
    # Get existing meta files
    meta_files = []
    for file in secretword_dir.glob("*-meta.json"):
        word = file.stem.replace("-meta", "")
        meta_files.append(word)
    
    meta_words = sorted(set(meta_files))
    print(f"📋 Found {len(meta_words)} existing meta files")
    
    # Find words needing meta generation
    words_needing_meta = [word for word in embedding_words if word not in meta_words]
    print(f"🎯 Words needing meta files: {len(words_needing_meta)}")
    
    if words_needing_meta:
        print(f"📝 First 10 words to process: {', '.join(words_needing_meta[:10])}")
        if len(words_needing_meta) > 10:
            print(f"    ... and {len(words_needing_meta) - 10} more")
    
    return words_needing_meta

def generate_meta_file(word):
    """Generate meta file for a single word"""
    print(f"\n{'='*60}")
    print(f"🤖 Processing: {word}")
    print(f"{'='*60}")
    
    try:
        # Set environment variables for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run([
            sys.executable, "scripts/processing/015_generate_word_meta.py", word
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        
        success = result.returncode == 0
        if success:
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to generate meta for '{word}': {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing '{word}': {e}")
        return False

def main():
    print("🚀 BATCH META FILE GENERATION")
    print("="*60)
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set the API key before running this script")
        return
    
    # Scan for words needing meta files
    words_needing_meta = scan_words_needing_meta()
    
    if not words_needing_meta:
        print("✅ All words with embeddings already have meta files!")
        return
    
    print(f"\n🎯 Starting batch processing for {len(words_needing_meta)} words...")
    print(f"⏱️ Estimated time: {len(words_needing_meta) * 0.5:.1f} minutes (30 seconds per word)")
    print(f"💰 Estimated cost: ${len(words_needing_meta) * 0.002:.3f} (OpenAI API calls)")
    
    # Confirm with user
    response = input(f"\nProceed with generating {len(words_needing_meta)} meta files? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return
    
    # Process each word
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, word in enumerate(words_needing_meta, 1):
        print(f"\n📊 Progress: {i}/{len(words_needing_meta)} ({i/len(words_needing_meta)*100:.1f}%)")
        
        if generate_meta_file(word):
            successful += 1
            print(f"✅ Successfully generated meta file for '{word}'")
        else:
            failed += 1
            print(f"❌ Failed to generate meta file for '{word}'")
        
        # Small delay to avoid rate limiting
        if i < len(words_needing_meta):  # Don't delay after the last word
            time.sleep(1)
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
    print(f"💰 Estimated cost: ${successful * 0.002:.3f}")
    
    if failed > 0:
        print(f"\n⚠️ {failed} words failed to process. Check the output above for details.")
    else:
        print(f"\n🎉 All meta files generated successfully!")

if __name__ == "__main__":
    main()

