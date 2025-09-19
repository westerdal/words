#!/usr/bin/env python3
"""
Consolidate individual embedding cache files into one master cache file.
This will read all individual JSON files in embedding_cache/ and combine them
into a single embeddings_cache_consolidated.json file.
"""

import os
import json
import sys
from datetime import datetime

def log(msg):
    """Print timestamped log message with immediate flush."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {msg}', flush=True)

def consolidate_embeddings():
    """Consolidate all individual embedding files into one master file."""
    
    log("=== EMBEDDING CACHE CONSOLIDATION ===")
    
    # Check if embedding_cache directory exists
    cache_dir = "embedding_cache"
    if not os.path.exists(cache_dir):
        log("❌ Error: embedding_cache directory not found")
        return False
    
    # Get all JSON files
    log("Scanning for embedding files...")
    json_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    total_files = len(json_files)
    log(f"📁 Found {total_files:,} embedding files to consolidate")
    
    if total_files == 0:
        log("❌ No embedding files found to consolidate")
        return False
    
    # Initialize consolidated cache
    consolidated_cache = {}
    processed = 0
    errors = 0
    
    log("🔄 Starting consolidation process...")
    
    # Process each file
    for filename in json_files:
        try:
            # Extract word from filename (remove .json extension)
            word = filename[:-5]  # Remove .json
            
            # Read the individual cache file
            filepath = os.path.join(cache_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            
            # Add to consolidated cache
            consolidated_cache[word] = embedding_data
            processed += 1
            
            # Progress update every 10,000 files
            if processed % 10000 == 0:
                log(f"📊 Progress: {processed:,}/{total_files:,} files ({processed/total_files*100:.1f}%)")
                
        except Exception as e:
            errors += 1
            log(f"⚠️  Error processing {filename}: {e}")
            continue
    
    log(f"✅ Processed {processed:,} files successfully")
    if errors > 0:
        log(f"⚠️  {errors} files had errors")
    
    # Write consolidated cache
    output_file = "embeddings_cache_consolidated.json"
    log(f"💾 Writing consolidated cache to {output_file}...")
    log("📝 Note: You should move this file to .env/embeddings.json to avoid syncing with Git")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_cache, f, separators=(',', ':'))
        
        # Get file size
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        
        log(f"🎉 Successfully created {output_file}")
        log(f"📏 File size: {file_size_mb:.1f} MB ({file_size:,} bytes)")
        log(f"📊 Contains embeddings for {len(consolidated_cache):,} words")
        
        return True
        
    except Exception as e:
        log(f"❌ Error writing consolidated file: {e}")
        return False

def main():
    """Main function."""
    log("Starting embedding consolidation process...")
    
    success = consolidate_embeddings()
    
    if success:
        log("🎉 Consolidation completed successfully!")
    else:
        log("❌ Consolidation failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
