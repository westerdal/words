#!/usr/bin/env python3
"""
Progress tracking and status updates for Semantic Rank processing
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .config import Config
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Config

class ProgressTracker:
    """Handles progress tracking, status updates, and checkpoint management"""
    
    def __init__(self, word: str, task_name: str, total_items: int = 0):
        self.word = word.lower().strip()
        self.task_name = task_name
        self.total_items = total_items
        self.current_item = 0
        
        # Timing
        self.start_time = datetime.now()
        self.last_status_time = self.start_time
        self.last_checkpoint_time = self.start_time
        
        # Progress file
        self.progress_file = Config.get_progress_filename(self.word)
        
        # Status tracking
        self.status_history = []
        
        # Ensure logs directory exists
        Config.ensure_directories()
        
        self._log(f"🚀 Starting {task_name} for '{word}' ({total_items:,} items)")
    
    def _log(self, message: str, force_flush: bool = True):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        
        if force_flush:
            sys.stdout.flush()
        
        # Add to status history
        self.status_history.append({
            'timestamp': timestamp,
            'message': message
        })
    
    def update(self, current: int, message: str = "", force_status: bool = False):
        """Update progress and show status if needed"""
        self.current_item = current
        
        now = datetime.now()
        time_since_status = (now - self.last_status_time).total_seconds()
        
        # Show status update if interval passed or forced
        if time_since_status >= Config.STATUS_UPDATE_INTERVAL or force_status:
            self._show_status(message)
            self.last_status_time = now
    
    def _show_status(self, extra_message: str = ""):
        """Show current progress status"""
        if self.total_items == 0:
            progress_pct = 0
            eta_str = "unknown"
        else:
            progress_pct = (self.current_item / self.total_items) * 100
            
            # Calculate ETA
            elapsed = datetime.now() - self.start_time
            if self.current_item > 0:
                rate = self.current_item / elapsed.total_seconds()
                remaining_items = self.total_items - self.current_item
                eta_seconds = remaining_items / rate if rate > 0 else 0
                eta_str = self._format_duration(eta_seconds)
            else:
                eta_str = "calculating..."
        
        # Format status message
        status_msg = f"🔄 {self.task_name.upper()} | {self.word} | "
        
        if self.total_items > 0:
            status_msg += f"Processing: {self.current_item:,}/{self.total_items:,} ({progress_pct:.1f}%) | ETA: {eta_str}"
        else:
            status_msg += f"Processing: {self.current_item:,} items"
        
        if extra_message:
            status_msg += f" | {extra_message}"
        
        self._log(status_msg)
    
    def checkpoint(self, data: Dict[str, Any], message: str = ""):
        """Save checkpoint data"""
        now = datetime.now()
        
        checkpoint_data = {
            'word': self.word,
            'task_name': self.task_name,
            'current_item': self.current_item,
            'total_items': self.total_items,
            'start_time': self.start_time.isoformat(),
            'checkpoint_time': now.isoformat(),
            'data': data
        }
        
        # Save to file
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            checkpoint_msg = f"💾 CHECKPOINT | {self.word} | Saved at {self.current_item:,} items"
            if message:
                checkpoint_msg += f" | {message}"
            
            self._log(checkpoint_msg)
            self.last_checkpoint_time = now
            
        except Exception as e:
            self._log(f"⚠️ WARNING | {self.word} | Failed to save checkpoint: {e}")
    
    def should_checkpoint(self) -> bool:
        """Check if it's time for a checkpoint"""
        return self.current_item > 0 and self.current_item % Config.CHECKPOINT_INTERVAL == 0
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data if it exists"""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate checkpoint
            if data.get('word') != self.word or data.get('task_name') != self.task_name:
                self._log(f"⚠️ WARNING | Checkpoint mismatch - ignoring old checkpoint")
                return None
            
            self.current_item = data.get('current_item', 0)
            
            checkpoint_time = data.get('checkpoint_time', '')
            self._log(f"🔁 RESUME | {self.word} | Continuing from {self.current_item:,} items (saved {checkpoint_time})")
            
            return data.get('data', {})
            
        except Exception as e:
            self._log(f"⚠️ WARNING | Failed to load checkpoint: {e}")
            return None
    
    def complete(self, final_message: str = ""):
        """Mark task as complete"""
        elapsed = datetime.now() - self.start_time
        elapsed_str = self._format_duration(elapsed.total_seconds())
        
        completion_msg = f"✅ COMPLETE | {self.word} | {self.task_name} finished in {elapsed_str}"
        if final_message:
            completion_msg += f" | {final_message}"
        
        self._log(completion_msg)
        
        # Clean up progress file
        if self.progress_file.exists():
            try:
                self.progress_file.unlink()
            except:
                pass
    
    def error(self, error_message: str):
        """Log error and preserve checkpoint"""
        elapsed = datetime.now() - self.start_time
        elapsed_str = self._format_duration(elapsed.total_seconds())
        
        error_msg = f"❌ ERROR | {self.word} | {self.task_name} failed after {elapsed_str} | {error_message}"
        self._log(error_msg)
    
    def warning(self, warning_message: str):
        """Log warning message"""
        warning_msg = f"⚠️ WARNING | {self.word} | {warning_message}"
        self._log(warning_msg)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        elapsed = datetime.now() - self.start_time
        
        stats = {
            'word': self.word,
            'task_name': self.task_name,
            'current_item': self.current_item,
            'total_items': self.total_items,
            'progress_pct': (self.current_item / self.total_items * 100) if self.total_items > 0 else 0,
            'elapsed_seconds': elapsed.total_seconds(),
            'elapsed_formatted': self._format_duration(elapsed.total_seconds()),
            'rate_per_second': self.current_item / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        }
        
        if self.total_items > 0 and self.current_item > 0:
            remaining_items = self.total_items - self.current_item
            rate = stats['rate_per_second']
            eta_seconds = remaining_items / rate if rate > 0 else 0
            stats['eta_seconds'] = eta_seconds
            stats['eta_formatted'] = self._format_duration(eta_seconds)
        
        return stats

# === CONVENIENCE FUNCTIONS ===

def create_tracker(word: str, task_name: str, total_items: int = 0) -> ProgressTracker:
    """Create a new progress tracker"""
    return ProgressTracker(word, task_name, total_items)

def quick_log(word: str, message: str):
    """Quick log message without full tracker"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    sys.stdout.flush()

if __name__ == "__main__":
    # Test progress tracker
    print("=== Progress Tracker Test ===")
    
    # Create test tracker
    tracker = create_tracker("forest", "TEST", 1000)
    
    # Simulate progress
    for i in range(0, 1001, 100):
        tracker.update(i, f"Processing item {i}")
        
        if tracker.should_checkpoint():
            tracker.checkpoint({'test_data': f'checkpoint_{i}'})
        
        time.sleep(0.1)  # Small delay to show timing
    
    tracker.complete("Test completed successfully")
    
    print("\n✅ Progress tracker test completed!")
