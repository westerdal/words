#!/usr/bin/env python3
"""
Batch Secret Word Generator - Process multiple secret words with queue management
"""

import json
import os
import time
from datetime import datetime
from pure_ai_generator import PureAISemanticRankGenerator

class BatchSecretWordGenerator:
    def __init__(self, batch_size=50):
        """Initialize the batch generator."""
        self.batch_size = batch_size
        self.queue_file = "secretword_queue.json"
        self.progress_file = "secretword_progress.json"
        self.failed_file = "secretword_failed.json"
        
        # Ensure secretword directory exists
        if not os.path.exists("secretword"):
            os.makedirs("secretword")
    
    def load_queue(self):
        """Load the processing queue from file."""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print("⚠️  Queue file corrupted or missing, starting fresh")
                return {"pending": [], "completed": [], "failed": []}
        return {"pending": [], "completed": [], "failed": []}
    
    def save_queue(self, queue_data):
        """Save the current queue state to file."""
        try:
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(queue_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save queue: {e}")
    
    def load_progress(self):
        """Load processing progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {
            "session_start": None,
            "last_processed": None,
            "total_processed": 0,
            "total_failed": 0,
            "current_word": None
        }
    
    def save_progress(self, progress_data):
        """Save processing progress to file."""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save progress: {e}")
    
    def add_words_to_queue(self, secret_words):
        """Add a list of secret words to the processing queue."""
        queue_data = self.load_queue()
        
        # Convert to lowercase and remove duplicates
        new_words = []
        existing_words = set(queue_data["pending"] + queue_data["completed"])
        
        for word in secret_words:
            word = word.strip().lower()
            if word and word not in existing_words:
                new_words.append(word)
                existing_words.add(word)
        
        queue_data["pending"].extend(new_words)
        self.save_queue(queue_data)
        
        print(f"✅ Added {len(new_words)} new words to queue")
        print(f"📊 Queue status: {len(queue_data['pending'])} pending, {len(queue_data['completed'])} completed, {len(queue_data['failed'])} failed")
        
        return len(new_words)
    
    def process_single_word(self, secret_word):
        """Process a single secret word and generate its CSV."""
        try:
            print(f"\n🔄 Processing: '{secret_word}'")
            start_time = time.time()
            
            # Check if file already exists
            output_file = f"secretword/secretword-{secret_word}.csv"
            if os.path.exists(output_file):
                print(f"⚠️  File already exists: {output_file}")
                file_size = os.path.getsize(output_file)
                if file_size > 1000000:  # If file is > 1MB, assume it's complete
                    print(f"✅ Skipping - file appears complete ({file_size:,} bytes)")
                    return True
                else:
                    print(f"🗑️  Removing incomplete file ({file_size:,} bytes)")
                    os.remove(output_file)
            
            # Generate the CSV
            generator = PureAISemanticRankGenerator(secret_word, self.batch_size)
            
            # Load words
            if not generator.load_words():
                print(f"❌ Failed to load word list for '{secret_word}'")
                return False
            
            # Compute rankings
            generator.compute_rankings()
            
            # Generate CSV
            result = generator.generate_csv()
            
            if result:
                elapsed_time = time.time() - start_time
                file_size = os.path.getsize(result)
                print(f"✅ Successfully generated {result}")
                print(f"⏱️  Time taken: {elapsed_time:.1f} seconds")
                print(f"💾 File size: {file_size:,} bytes")
                return True
            else:
                print(f"❌ Failed to generate CSV for '{secret_word}'")
                return False
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted for '{secret_word}'")
            raise
        except Exception as e:
            print(f"❌ Error processing '{secret_word}': {e}")
            return False
    
    def process_queue(self, max_words=None):
        """Process all words in the queue with graceful failure handling."""
        queue_data = self.load_queue()
        progress_data = self.load_progress()
        
        if not queue_data["pending"]:
            print("📭 No words in queue to process")
            return
        
        # Initialize session
        if not progress_data["session_start"]:
            progress_data["session_start"] = datetime.now().isoformat()
        
        print("=== Batch Secret Word Generator ===")
        print(f"📊 Queue status: {len(queue_data['pending'])} pending")
        print(f"🎯 Batch size: {self.batch_size}")
        if max_words:
            print(f"🔢 Max words this session: {max_words}")
        print()
        
        processed_this_session = 0
        
        try:
            while queue_data["pending"] and (max_words is None or processed_this_session < max_words):
                current_word = queue_data["pending"][0]
                progress_data["current_word"] = current_word
                self.save_progress(progress_data)
                
                print(f"📍 Progress: {processed_this_session + 1} of {min(len(queue_data['pending']), max_words or len(queue_data['pending']))}")
                
                success = self.process_single_word(current_word)
                
                # Remove from pending
                queue_data["pending"].remove(current_word)
                
                if success:
                    queue_data["completed"].append(current_word)
                    progress_data["total_processed"] += 1
                    print(f"✅ Added '{current_word}' to completed list")
                else:
                    queue_data["failed"].append({
                        "word": current_word,
                        "timestamp": datetime.now().isoformat(),
                        "error": "Processing failed"
                    })
                    progress_data["total_failed"] += 1
                    print(f"❌ Added '{current_word}' to failed list")
                
                # Update progress
                progress_data["last_processed"] = current_word
                progress_data["current_word"] = None
                processed_this_session += 1
                
                # Save state after each word
                self.save_queue(queue_data)
                self.save_progress(progress_data)
                
                print(f"💾 State saved - {len(queue_data['pending'])} remaining")
                
                # Small delay between words to be respectful
                time.sleep(1)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Batch processing interrupted!")
            print(f"📊 Session summary:")
            print(f"   - Processed this session: {processed_this_session}")
            print(f"   - Words remaining: {len(queue_data['pending'])}")
            print(f"   - Total completed: {len(queue_data['completed'])}")
            print(f"   - Total failed: {len(queue_data['failed'])}")
            
            # Save final state
            progress_data["current_word"] = None
            self.save_queue(queue_data)
            self.save_progress(progress_data)
            print("💾 State saved - you can resume processing later")
            return
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            progress_data["current_word"] = None
            self.save_queue(queue_data)
            self.save_progress(progress_data)
            return
        
        # Session completed successfully
        print(f"\n🎉 Batch processing completed!")
        print(f"📊 Final summary:")
        print(f"   - Processed this session: {processed_this_session}")
        print(f"   - Total completed: {len(queue_data['completed'])}")
        print(f"   - Total failed: {len(queue_data['failed'])}")
        print(f"   - Words remaining: {len(queue_data['pending'])}")
    
    def show_status(self):
        """Display current queue and progress status."""
        queue_data = self.load_queue()
        progress_data = self.load_progress()
        
        print("=== Batch Generator Status ===")
        print(f"📊 Queue Summary:")
        print(f"   - Pending: {len(queue_data['pending'])}")
        print(f"   - Completed: {len(queue_data['completed'])}")
        print(f"   - Failed: {len(queue_data['failed'])}")
        
        if progress_data["session_start"]:
            print(f"\n📈 Progress:")
            print(f"   - Session started: {progress_data['session_start']}")
            print(f"   - Last processed: {progress_data['last_processed']}")
            print(f"   - Total processed: {progress_data['total_processed']}")
            print(f"   - Total failed: {progress_data['total_failed']}")
            if progress_data["current_word"]:
                print(f"   - Currently processing: {progress_data['current_word']}")
        
        if queue_data["pending"]:
            print(f"\n📝 Next 10 words in queue:")
            for i, word in enumerate(queue_data["pending"][:10]):
                print(f"   {i+1:2d}. {word}")
            if len(queue_data["pending"]) > 10:
                print(f"   ... and {len(queue_data['pending']) - 10} more")
        
        if queue_data["failed"]:
            print(f"\n❌ Failed words:")
            for item in queue_data["failed"][-5:]:  # Show last 5 failures
                if isinstance(item, dict):
                    print(f"   - {item['word']} ({item.get('timestamp', 'unknown time')})")
                else:
                    print(f"   - {item}")
    
    def retry_failed(self):
        """Move failed words back to pending queue for retry."""
        queue_data = self.load_queue()
        
        if not queue_data["failed"]:
            print("📭 No failed words to retry")
            return
        
        failed_words = []
        for item in queue_data["failed"]:
            if isinstance(item, dict):
                failed_words.append(item["word"])
            else:
                failed_words.append(item)
        
        queue_data["pending"].extend(failed_words)
        queue_data["failed"] = []
        
        self.save_queue(queue_data)
        print(f"🔄 Moved {len(failed_words)} failed words back to pending queue")
    
    def clear_completed(self):
        """Clear the completed list (files remain, just clears tracking)."""
        queue_data = self.load_queue()
        cleared_count = len(queue_data["completed"])
        queue_data["completed"] = []
        self.save_queue(queue_data)
        print(f"🧹 Cleared {cleared_count} completed words from tracking")

def main():
    """Main function with command-line interface."""
    generator = BatchSecretWordGenerator()
    
    print("=== Batch Secret Word Generator ===")
    print("Commands:")
    print("  status  - Show current queue status")
    print("  add     - Add words to queue (interactive)")
    print("  process - Process all words in queue")
    print("  retry   - Retry failed words")
    print("  clear   - Clear completed list")
    print("  quit    - Exit")
    print()
    
    while True:
        try:
            command = input("Enter command (or 'quit'): ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'status':
                generator.show_status()
            elif command == 'add':
                print("Enter secret words (one per line, empty line to finish):")
                words = []
                while True:
                    word = input().strip()
                    if not word:
                        break
                    words.append(word)
                if words:
                    generator.add_words_to_queue(words)
                else:
                    print("No words entered")
            elif command == 'process':
                max_words_input = input("Max words to process (empty for all): ").strip()
                max_words = int(max_words_input) if max_words_input else None
                generator.process_queue(max_words)
            elif command == 'retry':
                generator.retry_failed()
            elif command == 'clear':
                generator.clear_completed()
            else:
                print("Unknown command. Try: status, add, process, retry, clear, quit")
            print()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
