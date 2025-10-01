#!/usr/bin/env python3
"""
Universal Word Game Master - A generic game master that works with any secret word
"""
import csv
import re
import os
from typing import Dict, List, Tuple, Optional

class UniversalGameMaster:
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        self.word_data: Dict[str, Dict] = {}
        self.dictionary: set = set()
        self.game_started = False
        self.guesses: List[Tuple[str, int, str]] = []  # (word, rank, clue)
        self.hints_used = 0
        self.max_hints = 10
        self.hint_history: List[Tuple[str, int, str]] = []  # Track given hints
        
    def load_word_data(self, csv_file: str):
        """Load the word data from CSV file"""
        try:
            if not os.path.exists(csv_file):
                print(f"❌ Game file not found: {csv_file}")
                return False
                
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row['word'].lower().strip()
                    self.word_data[word] = {
                        'rank': int(row['rank']),
                        'secret_word': row['secret_word'].lower().strip(),
                        'clue': row['clue'],
                        'connection_strength': row.get('connection_strength', 'medium')
                    }
            return True
        except Exception as e:
            print(f"Error loading word data: {e}")
            return False
        
    def load_dictionary(self, dict_file: str):
        """Load the scrabble dictionary"""
        try:
            with open(dict_file, 'r', encoding='utf-8') as f:
                self.dictionary = {line.strip().lower() for line in f}
            return True
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return False
        
    def convert_plural_to_singular(self, word: str) -> str:
        """Convert plural words to singular form - FIXED to not break singular words ending in 's'"""
        word = word.lower().strip()
        
        # If the word is already in the dictionary, don't convert it!
        # This prevents breaking singular words like 'grass', 'class', 'glass', etc.
        if word in self.dictionary:
            return word
        
        # Only try conversion if the word is NOT in dictionary
        # Common plural patterns
        if word.endswith('ies') and len(word) > 4:
            candidate = word[:-3] + 'y'
            if candidate in self.dictionary:
                return candidate
        elif word.endswith('es') and len(word) > 3:
            if word.endswith(('ches', 'shes', 'xes', 'zes')):
                candidate = word[:-2]
                if candidate in self.dictionary:
                    return candidate
            elif word.endswith('oes'):
                candidate = word[:-2]
                if candidate in self.dictionary:
                    return candidate
            else:
                candidate = word[:-1]
                if candidate in self.dictionary:
                    return candidate
        elif word.endswith('s') and len(word) > 2:
            candidate = word[:-1]
            if candidate in self.dictionary:
                return candidate
        
        # If no conversion worked, return original word
        return word
    
    def is_valid_word(self, word: str) -> bool:
        """Check if word is in the scrabble dictionary"""
        return word.lower().strip() in self.dictionary
    
    def get_csv_filename(self):
        """Get the CSV filename for the secret word"""
        return f"secretword/{self.secret_word}-secret.csv"
    
    def initialize_game(self):
        """Initialize the game with data loading"""
        print(f"🔄 Loading {self.secret_word.upper()} game data...")
        
        # Load word data
        csv_file = self.get_csv_filename()
        if not self.load_word_data(csv_file):
            return False
        print(f"✅ Loaded {len(self.word_data)} words for {self.secret_word.upper()}")
        
        # Load dictionary
        if not self.load_dictionary('data/enable2.txt'):
            return False
        print(f"✅ Loaded {len(self.dictionary)} dictionary words")
        
        return True
    
    def start_game(self):
        """Start the game and announce readiness"""
        self.game_started = True
        self.guesses = []
        self.hints_used = 0
        self.hint_history = []
        
        print("\n" + "="*60)
        print("🎮 UNIVERSAL WORD GUESSING GAME READY! 🎮")
        print("="*60)
        print("🎯 Try to guess the secret word!")
        print("🎲 Guess words and I'll show you how close you are!")
        print("📝 Each guess gets a rank and clue!")
        print("💡 Type 'hint' for help when you're stuck!")
        print("="*60 + "\n")
        
        return True
    
    def get_word_emoji(self, word: str) -> str:
        """Get appropriate emoji for the word"""
        emoji_map = {
            'cat': '🐱', 'dog': '🐕', 'fish': '🐟', 'bird': '🐦',
            'queen': '👑', 'king': '👑', 'wine': '🍷', 'art': '🎨',
            'rock': '🪨', 'forest': '🌲', 'tree': '🌳', 'flower': '🌸',
            'ocean': '🌊', 'mountain': '⛰️', 'sun': '☀️', 'moon': '🌙'
        }
        return emoji_map.get(word, '🎮')
    
    def get_best_rank(self) -> int:
        """Get the best (lowest) rank from guesses"""
        if not self.guesses:
            return float('inf')
        return min(guess[1] for guess in self.guesses)
    
    def get_hint_word(self, target_rank: int) -> Optional[Dict]:
        """Get a word at or near the target rank that hasn't been guessed"""
        # Find words around the target rank
        candidates = []
        for word, data in self.word_data.items():
            rank = data['rank']
            # Look for words within 5 ranks of target
            if abs(rank - target_rank) <= 5:
                # Skip if already guessed or given as hint
                already_used = (
                    any(guess[0] == word for guess in self.guesses) or
                    any(hint[0] == word for hint in self.hint_history) or
                    word == self.secret_word
                )
                if not already_used:
                    candidates.append((word, data))
        
        # Sort by closeness to target rank and prefer strong connections
        candidates.sort(key=lambda x: (
            abs(x[1]['rank'] - target_rank),
            0 if x[1].get('connection_strength') == 'strong' else 1
        ))
        
        return candidates[0] if candidates else None
    
    def generate_hint(self) -> dict:
        """Generate a hint based on current game state"""
        if self.hints_used >= self.max_hints:
            return {"error": f"🚫 No more hints available! You've used all {self.max_hints} hints."}
        
        best_rank = self.get_best_rank()
        
        if best_rank == float('inf'):
            return {"error": "❌ Make at least one guess before asking for a hint!"}
        
        # Determine hint target rank based on distance from goal
        if best_rank <= 50:
            # Very close - give word 10 ranks better
            target_rank = max(1, best_rank - 10)
            hint_level = "🔥 Hot"
        elif best_rank <= 100:
            # Close - give word 1 rank better  
            target_rank = max(1, best_rank - 1)
            hint_level = "🎯 Very Close"
        elif best_rank <= 1000:
            # Medium distance - give word 25 ranks better
            target_rank = max(1, best_rank - 25)
            hint_level = "🎲 Getting Warmer"
        else:
            # Far away - give word 500 ranks better
            target_rank = max(1, best_rank - 500)
            hint_level = "🧭 Direction"
        
        # Get hint word
        hint_candidate = self.get_hint_word(target_rank)
        if not hint_candidate:
            return {"error": "❌ No suitable hint available at this time. Keep guessing!"}
        
        hint_word, hint_data = hint_candidate
        self.hints_used += 1
        
        # Add to hint history
        self.hint_history.append((hint_word, hint_data['rank'], hint_data['clue']))
        
        return {
            "success": True,
            "word": hint_word,
            "rank": hint_data['rank'],
            "clue": hint_data['clue'],
            "hint_level": hint_level,
            "hints_used": self.hints_used,
            "hints_remaining": self.max_hints - self.hints_used
        }
    
    def process_guess(self, guess: str) -> dict:
        """Process a player's guess and return result"""
        if not self.game_started:
            return {"error": "Game not started! Please start the game first."}
        
        # Check for hint request
        if guess.lower().strip() == 'hint':
            return self.generate_hint()
            
        # Convert to lowercase and strip
        original_guess = guess.strip()
        guess = guess.lower().strip()
        
        # Convert plural to singular if needed
        singular_guess = self.convert_plural_to_singular(guess)
        
        # Check if word is in dictionary
        if not self.is_valid_word(singular_guess):
            return {"error": f"The word '{original_guess}' is not in the scrabble dictionary"}
        
        # Check if word is in our game data
        if singular_guess not in self.word_data:
            return {"error": f"The word '{original_guess}' is not related to the secret word"}
        
        # Get word info
        word_info = self.word_data[singular_guess]
        rank = word_info['rank']
        clue = word_info['clue'] if word_info['clue'].strip() else "Not a close association"
        
        # Check if it's the secret word
        if singular_guess == self.secret_word:
            return {
                "winner": True,
                "word": singular_guess,
                "rank": rank,
                "clue": clue
            }
        
        # Add to guesses if not already guessed
        if not any(g[0] == singular_guess for g in self.guesses):
            self.guesses.append((singular_guess, rank, clue))
        
        return {
            "success": True,
            "word": singular_guess,
            "rank": rank,
            "clue": clue,
            "total_guesses": len(self.guesses)
        }
    
    def display_result(self, result: dict):
        """Display the result of a guess or hint"""
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        # Handle hint display
        if result.get("hint_level"):
            print(f"\n💡 {result['hint_level']} HINT #{result['hints_used']} 💡")
            print("="*60)
            print(f"Try: {result['word'].upper()}")
            print(f"Rank: #{result['rank']}")
            print(f"Clue: {result['clue']}")
            print(f"Hints remaining: {result['hints_remaining']}")
            print("="*60)
            return
        
        # Handle winner display
        if result.get("winner"):
            emoji = self.get_word_emoji(self.secret_word)
            print("\n" + emoji * 20)
            print(f"🏆 CONGRATULATIONS! YOU FOUND {self.secret_word.upper()}! 🏆")
            print(emoji * 20)
        
        print("\n" + "="*60)
        if result.get("winner"):
            print(f"{self.get_word_emoji(self.secret_word)} SECRET WORD FOUND! {self.get_word_emoji(self.secret_word)}")
        else:
            print("📝 GUESS RESULT")
        print("="*60)
        print(f"Word: {result['word'].upper()}")
        print(f"Rank: #{result['rank']}")
        print(f"Clue: {result['clue']}")
        print("="*60)
    
    def display_leaderboard(self):
        """Display the leaderboard of all guesses"""
        if not self.guesses and not self.hint_history:
            return
            
        print(f"\n🎯 LEADERBOARD (Best Guesses) 🎯")
        print("-" * 60)
        
        # Combine guesses and hints, sort by rank
        all_entries = []
        
        # Add guesses
        for word, rank, clue in self.guesses:
            all_entries.append((word, rank, clue, "GUESS"))
        
        # Add hints
        for word, rank, clue in self.hint_history:
            all_entries.append((word, rank, clue, "HINT"))
        
        # Sort by rank (lower rank = better)
        all_entries.sort(key=lambda x: x[1])
        
        # Show only top 25 entries to avoid clutter
        display_entries = all_entries[:25]
        
        for i, (word, rank, clue, entry_type) in enumerate(display_entries, 1):
            # Truncate clue to fit nicely
            clue_display = clue[:30] + "..." if len(clue) > 30 else clue
            type_icon = "💡" if entry_type == "HINT" else "🎯"
            print(f"{i:2d}. #{rank:3d} - {word.upper():12s} {type_icon} | {clue_display}")
        
        # Show if there are more entries
        if len(all_entries) > 25:
            remaining = len(all_entries) - 25
            print(f"    ... and {remaining} more entries (showing top 25)")
        
        
        print("-" * 60)
        print(f"Total guesses: {len(self.guesses)} | Hints used: {self.hints_used}/{self.max_hints}")
        if all_entries:
            best_rank = min(entry[1] for entry in all_entries)
            print(f"Best rank so far: #{best_rank}")
        print()

def create_game_master(secret_word: str) -> UniversalGameMaster:
    """Factory function to create a game master for any word"""
    return UniversalGameMaster(secret_word)

if __name__ == "__main__":
    # Example usage
    word = input("Enter secret word: ").strip()
    if word:
        game = create_game_master(word)
        if game.initialize_game():
            game.start_game()
            print(f"🎮 {word.upper()} game ready!")
        else:
            print("❌ Failed to setup game!")
