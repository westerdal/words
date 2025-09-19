#!/usr/bin/env python3
"""
Interactive Word Game Master - A game master that can be controlled programmatically
"""
import csv
import re
from typing import Dict, List, Tuple, Optional

class InteractiveGameMaster:
    def __init__(self):
        self.word_data: Dict[str, Dict] = {}
        self.dictionary: set = set()
        self.game_started = False
        self.secret_word = ""
        self.guesses: List[Tuple[str, int, str]] = []  # (word, rank, clue)
        
    def load_word_data(self, csv_file: str):
        """Load the word data from CSV file"""
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row['word'].lower().strip()
                    self.word_data[word] = {
                        'rank': int(row['rank']),
                        'secret_word': row['secret_word'].lower().strip(),
                        'clue': row['clue'],
                        'connection_strength': row['connection_strength']
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
        """Convert plural words to singular form"""
        word = word.lower().strip()
        
        # Common plural patterns
        if word.endswith('ies') and len(word) > 4:
            # flies -> fly, tries -> try
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            # boxes -> box, dishes -> dish
            if word.endswith(('ches', 'shes', 'xes', 'zes')):
                return word[:-2]
            # heroes -> hero
            elif word.endswith('oes'):
                return word[:-2]
            # else just remove 's'
            else:
                return word[:-1]
        elif word.endswith('s') and len(word) > 2:
            # cats -> cat, dogs -> dog
            return word[:-1]
        
        return word
    
    def is_valid_word(self, word: str) -> bool:
        """Check if word is in the scrabble dictionary"""
        return word.lower().strip() in self.dictionary
    
    def initialize_game(self):
        """Initialize the game with data loading"""
        print("🔄 Loading game data...")
        
        # Load word data
        if not self.load_word_data('secretword/secretword-easy-animals-cat.csv'):
            return False
        print(f"✅ Loaded {len(self.word_data)} words")
        
        # Load dictionary
        if not self.load_dictionary('data/enable2.txt'):
            return False
        print(f"✅ Loaded {len(self.dictionary)} dictionary words")
        
        # Get the secret word from the data
        if self.word_data:
            self.secret_word = list(self.word_data.values())[0]['secret_word']
        
        return True
    
    def start_game(self):
        """Start the game and announce readiness"""
        self.game_started = True
        self.guesses = []
        
        print("\n" + "="*60)
        print("🎮 WORD GUESSING GAME READY! 🎮")
        print("="*60)
        print("🎯 Try to guess the secret word!")
        print("🎲 Guess words and I'll show you how close you are!")
        print("📝 Each guess gets a rank and clue!")
        print("="*60 + "\n")
        
        return True
    
    def process_guess(self, guess: str) -> dict:
        """Process a player's guess and return result"""
        if not self.game_started:
            return {"error": "Game not started! Please start the game first."}
            
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
        """Display the result of a guess"""
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        if result.get("winner"):
            print("\n" + "🎉" * 20)
            print("🏆 CONGRATULATIONS! YOU WON! 🏆")
            print("🎉" * 20)
        
        print("\n" + "="*60)
        if result.get("winner"):
            print("🎯 SECRET WORD FOUND! 🎯")
        else:
            print("📝 GUESS RESULT")
        print("="*60)
        print(f"Word: {result['word'].upper()}")
        print(f"Rank: #{result['rank']}")
        print(f"Clue: {result['clue']}")
        print("="*60)
    
    def display_leaderboard(self):
        """Display the leaderboard of all guesses"""
        if not self.guesses:
            return
            
        print("\n🏆 LEADERBOARD (Best Guesses) 🏆")
        print("-" * 60)
        
        # Sort by rank (lower rank = better)
        sorted_guesses = sorted(self.guesses, key=lambda x: x[1])
        
        for i, (word, rank, clue) in enumerate(sorted_guesses, 1):
            # Truncate clue to fit nicely
            clue_display = clue[:35] + "..." if len(clue) > 35 else clue
            print(f"{i:2d}. #{rank:3d} - {word.upper():12s} | {clue_display}")
        
        print("-" * 60)
        print(f"Total guesses: {len(self.guesses)}")
        if sorted_guesses:
            print(f"Best rank so far: #{sorted_guesses[0][1]}")
        print()

# Create the game master instance
game_master = InteractiveGameMaster()

def setup_game():
    """Setup and start the game"""
    if game_master.initialize_game():
        game_master.start_game()
        return True
    return False

def make_guess(word: str):
    """Make a guess in the game"""
    result = game_master.process_guess(word)
    game_master.display_result(result)
    
    if result.get("success") and not result.get("winner"):
        game_master.display_leaderboard()
    
    return result.get("winner", False)

if __name__ == "__main__":
    if setup_game():
        print("Game ready! Use make_guess('your_word') to play!")
    else:
        print("Failed to setup game!")
