#!/usr/bin/env python3
"""
Robust Batch Secret Word Generator - With timeouts, status updates, and better error handling
"""

import json
import os
import time
import signal
import threading
from datetime import datetime
from semantic_embedding_generator import SemanticEmbeddingGenerator

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

class RobustBatchGenerator:
    def __init__(self, batch_size=50):
        """Initialize the robust batch generator."""
        self.batch_size = batch_size
        self.queue_file = "secretword_queue.json"
        self.progress_file = "secretword_progress.json"
        self.failed_file = "secretword_failed.json"
        
        # Status tracking
        self.current_status = "Idle"
        self.current_word = None
        self.start_time = None
        self.last_status_update = 0
        self.status_interval = 30  # 30 seconds
        
        # Timeout settings
        self.word_timeout = 600  # 10 minutes per word max
        self.api_timeout = 30    # 30 seconds per API call max
        
        # Ensure secretword directory exists
        if not os.path.exists("secretword"):
            os.makedirs("secretword")
    
    def timeout_handler(self, signum, frame):
        """Handle timeout signals."""
        raise TimeoutError("Operation timed out")
    
    def update_status(self, status, word=None):
        """Update current status and word being processed."""
        self.current_status = status
        self.current_word = word
        self.last_status_update = time.time()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        if word:
            print(f"[{timestamp}] 📍 Status: {status} - '{word}'")
        else:
            print(f"[{timestamp}] 📍 Status: {status}")
    
    def print_periodic_status(self):
        """Print status update if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_status_update >= self.status_interval:
            elapsed = int(current_time - (self.start_time or current_time))
            elapsed_str = f"{elapsed//60}m {elapsed%60}s" if elapsed >= 60 else f"{elapsed}s"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            if self.current_word:
                print(f"[{timestamp}] ⏰ Still working: {self.current_status} - '{self.current_word}' (elapsed: {elapsed_str})")
            else:
                print(f"[{timestamp}] ⏰ Status: {self.current_status} (elapsed: {elapsed_str})")
            
            self.last_status_update = current_time
    
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
            if word and word not in existing_words and word.isalpha():  # Only alphabetic words
                new_words.append(word)
                existing_words.add(word)
        
        queue_data["pending"].extend(new_words)
        self.save_queue(queue_data)
        
        print(f"✅ Added {len(new_words)} new words to queue")
        print(f"📊 Queue status: {len(queue_data['pending'])} pending, {len(queue_data['completed'])} completed, {len(queue_data['failed'])} failed")
        
        return len(new_words)
    
    def process_single_word_with_timeout(self, secret_word):
        """Process a single word with timeout protection."""
        result = {"success": False, "error": None}
        
        def target():
            try:
                result["success"] = self.process_single_word_internal(secret_word)
            except Exception as e:
                result["error"] = str(e)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait with timeout
        thread.join(timeout=self.word_timeout)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            result["error"] = f"Timeout after {self.word_timeout} seconds"
            print(f"⏰ Timeout processing '{secret_word}' - moving to next word")
            return False
        
        if result["error"]:
            print(f"❌ Error processing '{secret_word}': {result['error']}")
            return False
        
        return result["success"]
    
    def process_single_word_internal(self, secret_word):
        """Internal method to process a single secret word."""
        try:
            self.update_status("Checking existing file", secret_word)
            
            # Check if file already exists
            output_file = f"secretword/secretword-{secret_word}.csv"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                if file_size > 1000000:  # If file is > 1MB, assume it's complete
                    print(f"✅ Skipping - file already exists and appears complete ({file_size:,} bytes)")
                    return True
                else:
                    print(f"🗑️  Removing incomplete file ({file_size:,} bytes)")
                    os.remove(output_file)
            
            self.update_status("Initializing generator", secret_word)
            
            # Create generator with timeout-aware modifications
            generator = TimeoutAwareGenerator(secret_word, self.batch_size, self.api_timeout)
            
            self.update_status("Loading word list", secret_word)
            
            # Load words
            if not generator.load_words():
                print(f"❌ Failed to load word list for '{secret_word}'")
                return False
            
            self.update_status("Computing rankings", secret_word)
            
            # Compute semantic rankings using cached embeddings (much faster!)
            if not generator.compute_semantic_rankings_from_cache():
                print("⚠️  Failed to use cached embeddings, falling back to API...")
                if not generator.compute_semantic_rankings():
                    raise Exception("Failed to compute semantic rankings")
            
            self.update_status("Generating AI clues", secret_word)
            
            # Generate CSV with progress updates
            result = generator.generate_csv_with_progress(self.print_periodic_status)
            
            if result:
                file_size = os.path.getsize(result)
                self.update_status("Completed successfully", secret_word)
                print(f"✅ Successfully generated {result} ({file_size:,} bytes)")
                return True
            else:
                print(f"❌ Failed to generate CSV for '{secret_word}'")
                return False
                
        except TimeoutError:
            print(f"⏰ Timeout processing '{secret_word}'")
            return False
        except Exception as e:
            print(f"❌ Error processing '{secret_word}': {e}")
            return False
    
    def process_queue(self, max_words=None):
        """Process all words in the queue with robust error handling."""
        queue_data = self.load_queue()
        progress_data = self.load_progress()
        
        if not queue_data["pending"]:
            print("📭 No words in queue to process")
            return
        
        # Initialize session
        self.start_time = time.time()
        if not progress_data["session_start"]:
            progress_data["session_start"] = datetime.now().isoformat()
        
        print("=== Robust Batch Secret Word Generator ===")
        print(f"📊 Queue status: {len(queue_data['pending'])} pending")
        print(f"🎯 Batch size: {self.batch_size}")
        print(f"⏰ Word timeout: {self.word_timeout} seconds")
        print(f"📡 API timeout: {self.api_timeout} seconds")
        if max_words:
            print(f"🔢 Max words this session: {max_words}")
        print()
        
        processed_this_session = 0
        
        try:
            while queue_data["pending"] and (max_words is None or processed_this_session < max_words):
                current_word = queue_data["pending"][0]
                progress_data["current_word"] = current_word
                self.save_progress(progress_data)
                
                remaining = len(queue_data["pending"])
                total_to_process = min(remaining, max_words or remaining)
                
                print(f"\n📍 Progress: {processed_this_session + 1} of {total_to_process} ({remaining} total remaining)")
                
                # Process with timeout protection
                word_start_time = time.time()
                success = self.process_single_word_with_timeout(current_word)
                word_elapsed = time.time() - word_start_time
                
                # Remove from pending
                queue_data["pending"].remove(current_word)
                
                if success:
                    queue_data["completed"].append(current_word)
                    progress_data["total_processed"] += 1
                    print(f"✅ Completed '{current_word}' in {word_elapsed:.1f}s")
                else:
                    queue_data["failed"].append({
                        "word": current_word,
                        "timestamp": datetime.now().isoformat(),
                        "error": "Processing failed or timed out",
                        "elapsed_seconds": word_elapsed
                    })
                    progress_data["total_failed"] += 1
                    print(f"❌ Failed '{current_word}' after {word_elapsed:.1f}s")
                
                # Update progress
                progress_data["last_processed"] = current_word
                progress_data["current_word"] = None
                processed_this_session += 1
                
                # Save state after each word
                self.save_queue(queue_data)
                self.save_progress(progress_data)
                
                remaining_after = len(queue_data["pending"])
                print(f"💾 State saved - {remaining_after} remaining in queue")
                
                # Brief pause between words
                if remaining_after > 0:
                    self.update_status("Preparing next word")
                    time.sleep(2)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Batch processing interrupted!")
            self.print_session_summary(processed_this_session, queue_data)
            
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
        self.print_session_summary(processed_this_session, queue_data)
    
    def print_session_summary(self, processed_this_session, queue_data):
        """Print session summary statistics."""
        total_elapsed = time.time() - (self.start_time or time.time())
        elapsed_str = f"{int(total_elapsed//60)}m {int(total_elapsed%60)}s"
        
        print(f"📊 Session Summary:")
        print(f"   - Processed this session: {processed_this_session}")
        print(f"   - Total completed: {len(queue_data['completed'])}")
        print(f"   - Total failed: {len(queue_data['failed'])}")
        print(f"   - Words remaining: {len(queue_data['pending'])}")
        print(f"   - Total time: {elapsed_str}")
        if processed_this_session > 0:
            avg_time = total_elapsed / processed_this_session
            print(f"   - Average per word: {avg_time:.1f}s")
    
    def show_status(self):
        """Display current queue and progress status."""
        queue_data = self.load_queue()
        progress_data = self.load_progress()
        
        print("=== Robust Batch Generator Status ===")
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
            print(f"\n❌ Recent failures:")
            for item in queue_data["failed"][-5:]:  # Show last 5 failures
                if isinstance(item, dict):
                    elapsed = item.get('elapsed_seconds', 0)
                    print(f"   - {item['word']} (failed after {elapsed:.1f}s)")
                else:
                    print(f"   - {item}")


class TimeoutAwareGenerator(SemanticEmbeddingGenerator):
    """Enhanced generator with timeout awareness and progress reporting."""
    
    def __init__(self, secret_word, batch_size, api_timeout):
        super().__init__(secret_word, batch_size)
        self.api_timeout = api_timeout
        self.progress_callback = None
    
    def generate_csv_with_progress(self, progress_callback):
        """Generate CSV with progress reporting."""
        self.progress_callback = progress_callback
        return self.generate_csv()
    
    def generate_batched_ai_clues(self, word_batch):
        """Generate clues with timeout protection."""
        if not self.use_ai:
            return {}
        
        try:
            # Call progress callback
            if self.progress_callback:
                self.progress_callback()
            
            # Filter out words that should get static clues
            ai_words = [(word, rank) for word, rank in word_batch if rank <= self.ai_cutoff_rank and rank > 1]
            
            if not ai_words:
                return {}
            
            # Create batch prompt (same as before)
            word_descriptions = []
            for word, rank in ai_words:
                word_descriptions.append(f'"{word}" (rank {rank})')
            
            prompt = f"""You are writing clues for a word guessing game. The secret word is "{self.secret_word}".

For each word below, write a SHORT clue (maximum 5 words) that hints at the relationship between that word and "{self.secret_word}". Use "*" as a placeholder for "{self.secret_word}".

Be creative and specific. Think about:
- How the word relates to {self.secret_word}
- What connection players might recognize
- Keep it under 5 words
- Use * instead of "{self.secret_word}"

Words to create clues for:
{chr(10).join(word_descriptions)}

Respond ONLY with valid JSON in this exact format:
{{"word1": "clue text", "word2": "clue text"}}

Example good clues:
- "Orbits around *"
- "Red neighbor of *"
- "Tool to observe *"
- "Force that holds *"
"""

            # Set timeout for API call
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("API call timed out")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.api_timeout)
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You write concise word game clues. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.8
                )
                
                # Cancel timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                content = response.choices[0].message.content.strip()
                
                # Clean up JSON response
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```', '').strip()
                
                try:
                    import json
                    clue_dict = json.loads(content)
                    
                    # Clean and validate clues
                    cleaned_clues = {}
                    for word, clue in clue_dict.items():
                        if isinstance(clue, str):
                            clue = clue.strip().strip('"').strip("'")
                            # Ensure max 5 words
                            words_in_clue = clue.split()
                            if len(words_in_clue) > 5:
                                clue = ' '.join(words_in_clue[:5])
                            cleaned_clues[word.lower()] = clue
                    
                    return cleaned_clues
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parsing failed: {e}")
                    return {}
            
            except TimeoutError:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"⏰ API call timed out after {self.api_timeout}s")
                return {}
                
        except Exception as e:
            print(f"⚠️  AI batch request failed: {e}")
            return {}


def main():
    """Main function with command-line interface."""
    generator = RobustBatchGenerator()
    
    print("=== Robust Batch Secret Word Generator ===")
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
                queue_data = generator.load_queue()
                if not queue_data["failed"]:
                    print("📭 No failed words to retry")
                else:
                    failed_words = []
                    for item in queue_data["failed"]:
                        if isinstance(item, dict):
                            failed_words.append(item["word"])
                        else:
                            failed_words.append(item)
                    
                    queue_data["pending"].extend(failed_words)
                    queue_data["failed"] = []
                    
                    generator.save_queue(queue_data)
                    print(f"🔄 Moved {len(failed_words)} failed words back to pending queue")
            elif command == 'clear':
                queue_data = generator.load_queue()
                cleared_count = len(queue_data["completed"])
                queue_data["completed"] = []
                generator.save_queue(queue_data)
                print(f"🧹 Cleared {cleared_count} completed words from tracking")
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
