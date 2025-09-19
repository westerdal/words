#!/usr/bin/env python3
"""
Robust Batch Secret Word Generator with Dynamic AI Cutoff
Integrates relationship strength detection to optimize API usage across multiple words
"""

import json
import os
import time
import threading
from datetime import datetime
from semantic_embedding_generator_dynamic import DynamicSemanticEmbeddingGenerator

class RobustDynamicBatchGenerator:
    def __init__(self):
        """Initialize the robust dynamic batch generator."""
        self.queue_file = "secretword_queue.json"
        self.progress_file = "secretword_progress.json"
        self.failed_file = "secretword_failed.json"
        
        # Status tracking
        self.current_status = "Idle"
        self.current_word = None
        self.start_time = None
        self.last_status_update = 0
        self.status_interval = 30  # 30 seconds
        
        # Dynamic cutoff settings
        self.consecutive_weak_threshold = 5
        
        # Ensure secretword directory exists
        if not os.path.exists("secretword"):
            os.makedirs("secretword")
    
    def update_status(self, status, word=None):
        """Update current status and word being processed."""
        self.current_status = status
        self.current_word = word
        self.last_status_update = time.time()
        
        # Print status update
        timestamp = datetime.now().strftime("%H:%M:%S")
        if word:
            print(f"[{timestamp}] 📊 Status: {status} - Word: {word}")
        else:
            print(f"[{timestamp}] 📊 Status: {status}")
    
    def start_status_monitor(self):
        """Start background thread to print status updates every 30 seconds."""
        def status_monitor():
            while True:
                time.sleep(self.status_interval)
                if self.current_status != "Idle":
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if self.current_word:
                        print(f"[{timestamp}] ⏰ Still processing: {self.current_word} - Status: {self.current_status} (Elapsed: {elapsed:.0f}s)")
                    else:
                        print(f"[{timestamp}] ⏰ Status: {self.current_status} (Elapsed: {elapsed:.0f}s)")
        
        monitor_thread = threading.Thread(target=status_monitor, daemon=True)
        monitor_thread.start()
    
    def load_queue(self):
        """Load the processing queue from file."""
        if not os.path.exists(self.queue_file):
            return []
        
        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading queue: {e}")
            return []
    
    def save_queue(self, queue):
        """Save the processing queue to file."""
        try:
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving queue: {e}")
    
    def load_progress(self):
        """Load progress tracking from file."""
        if not os.path.exists(self.progress_file):
            return {}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading progress: {e}")
            return {}
    
    def save_progress(self, progress):
        """Save progress tracking to file."""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving progress: {e}")
    
    def load_failed(self):
        """Load failed words from file."""
        if not os.path.exists(self.failed_file):
            return []
        
        try:
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading failed list: {e}")
            return []
    
    def save_failed(self, failed):
        """Save failed words to file."""
        try:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving failed list: {e}")
    
    def _process_word(self, word_entry):
        """Process a single word with dynamic cutoff system."""
        difficulty = word_entry['difficulty']
        category = word_entry['category']
        word = word_entry['word']
        
        self.update_status("Initializing generator", word)
        
        # Check if already processed
        csv_filename = f"secretword/secretword-{difficulty}-{category}-{word}.csv"
        if os.path.exists(csv_filename):
            print(f"✅ Already exists: {csv_filename} - skipping")
            return True
        
        # Create generator with dynamic cutoff
        generator = DynamicSemanticEmbeddingGenerator(
            word, 
            consecutive_weak_threshold=self.consecutive_weak_threshold
        )
        
        self.update_status("Loading word list", word)
        
        # Load words
        if not generator.load_words("data/enable2.txt"):
            print(f"❌ Failed to load word list for '{word}'")
            return False
        
        self.update_status("Computing semantic rankings", word)
        
        # Compute semantic rankings using cached embeddings
        if not generator.compute_semantic_rankings_from_cache(".env/embeddings2.json"):
            print(f"❌ Failed to compute semantic rankings for '{word}'")
            return False
        
        self.update_status("Generating CSV with dynamic cutoff", word)
        
        # Generate CSV with dynamic cutoff
        if not generator.generate_csv_with_dynamic_cutoff(csv_filename):
            print(f"❌ Failed to generate CSV for '{word}'")
            return False
        
        # Print dynamic cutoff statistics
        stats = generator.get_stats()
        print(f"\n📊 Dynamic Cutoff Results for '{word}':")
        print(f"   AI calls made: {stats['total_ai_calls']:,}")
        print(f"   Cutoff triggered: {'Yes' if stats['ai_cutoff_reached'] else 'No'}")
        if stats['cutoff_rank']:
            saved_calls = stats['total_words'] - stats['cutoff_rank']
            savings_percent = (saved_calls / stats['total_words']) * 100
            print(f"   API calls saved: ~{saved_calls:,} ({savings_percent:.1f}%)")
        
        print(f"✅ Successfully processed '{word}' with dynamic cutoff")
        return True
    
    def process_queue(self):
        """Process all words in the queue with dynamic cutoff."""
        print("=== Robust Dynamic Batch Generator ===")
        print(f"Dynamic cutoff threshold: {self.consecutive_weak_threshold} consecutive weak relationships")
        
        # Start status monitor
        self.start_status_monitor()
        
        # Load queue and progress
        queue = self.load_queue()
        progress = self.load_progress()
        failed = self.load_failed()
        
        if not queue:
            print("❌ No words in queue to process")
            return False
        
        print(f"📋 Found {len(queue)} words in queue")
        
        # Filter out already processed words
        remaining_queue = []
        for word_entry in queue:
            word_id = f"{word_entry['difficulty']}-{word_entry['category']}-{word_entry['word']}"
            if word_id not in progress:
                remaining_queue.append(word_entry)
        
        if not remaining_queue:
            print("✅ All words in queue have been processed")
            return True
        
        print(f"🔄 Processing {len(remaining_queue)} remaining words...")
        
        self.start_time = time.time()
        successful = 0
        failed_count = 0
        
        for i, word_entry in enumerate(remaining_queue, 1):
            word = word_entry['word']
            word_id = f"{word_entry['difficulty']}-{word_entry['category']}-{word}"
            
            print(f"\n--- Processing {i}/{len(remaining_queue)}: {word} ---")
            
            try:
                success = self._process_word(word_entry)
                
                if success:
                    # Mark as completed
                    progress[word_id] = {
                        'status': 'completed',
                        'timestamp': datetime.now().isoformat(),
                        'dynamic_cutoff_used': True
                    }
                    successful += 1
                else:
                    # Mark as failed
                    progress[word_id] = {
                        'status': 'failed',
                        'timestamp': datetime.now().isoformat(),
                        'error': 'Processing failed'
                    }
                    failed.append(word_entry)
                    failed_count += 1
                
                # Save progress after each word
                self.save_progress(progress)
                if failed_count > 0:
                    self.save_failed(failed)
                
            except Exception as e:
                print(f"❌ Unexpected error processing '{word}': {e}")
                
                # Mark as failed
                progress[word_id] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                failed.append(word_entry)
                failed_count += 1
                
                self.save_progress(progress)
                self.save_failed(failed)
        
        # Final results
        self.update_status("Completed")
        
        total_time = time.time() - self.start_time
        print(f"\n=== Dynamic Batch Processing Complete ===")
        print(f"✅ Successfully processed: {successful}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"⚡ Average time per word: {total_time/len(remaining_queue):.1f} seconds")
        
        if successful > 0:
            print(f"💰 Dynamic cutoff system optimized API usage across all words!")
            print(f"📁 CSV files saved in secretword/ directory")
        
        return successful > 0

def main():
    """Main function"""
    generator = RobustDynamicBatchGenerator()
    success = generator.process_queue()
    
    if success:
        print("\n🎉 Dynamic batch processing completed successfully!")
    else:
        print("\n❌ Dynamic batch processing failed!")
    
    return success

if __name__ == "__main__":
    main()
