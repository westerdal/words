#!/usr/bin/env python3
"""
Word Game Master - A game where players guess words and get ranked clues
"""
import csv
import re
from typing import Dict, List, Tuple, Optional

class WordGameMaster:
    def __init__(self):
        self.word_data: Dict[str, Dict] = {}
        self.dictionary: set = set()
        self.game_started = False
        self.secret_word = ""
        self.guesses: List[Tuple[str, int, str]] = []  # (word, rank, clue)
        
    def load_word_data(self, csv_file: str):
        """Load the word data from CSV file"""
        print("Loading word data...")
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
        print(f"Loaded {len(self.word_data)} words")
        
    def load_dictionary(self, dict_file: str):
        """Load the scrabble dictionary"""
        print("Loading scrabble dictionary...")
        with open(dict_file, 'r', encoding='utf-8') as f:
            self.dictionary = {line.strip().lower() for line in f}
        print(f"Loaded {len(self.dictionary)} dictionary words")
        
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
    
    def start_game(self):
        """Start the game and announce readiness"""
        if not self.word_data:
            print("❌ Error: Word data not loaded!")
            return False
            
        if not self.dictionary:
            print("❌ Error: Dictionary not loaded!")
            return False
            
        # Get the secret word from the data
        self.secret_word = list(self.word_data.values())[0]['secret_word']
        self.game_started = True
        self.guesses = []
        
        print("\n" + "="*60)
        print("🎮 WORD GUESSING GAME READY! 🎮")
        print("="*60)
        print("🎯 Try to guess the secret word!")
        print("🎲 Guess words and I'll tell you how close you are!")
        print("📝 Type your guess and press Enter.")
        print("="*60 + "\n")
        
        return True
    
    def make_guess(self, guess: str) -> bool:
        """Process a player's guess"""
        if not self.game_started:
            print("❌ Game not started! Please start the game first.")
            return False
            
        # Convert to lowercase and strip
        original_guess = guess.strip()
        guess = guess.lower().strip()
        
        # Convert plural to singular if needed
        singular_guess = self.convert_plural_to_singular(guess)
        
        # Check if word is in dictionary
        if not self.is_valid_word(singular_guess):
            print(f"❌ The word '{original_guess}' is not in the scrabble dictionary")
            return False
        
        # Check if word is in our game data
        if singular_guess not in self.word_data:
            print(f"❌ The word '{original_guess}' is not related to the secret word")
            return False
        
        # Get word info
        word_info = self.word_data[singular_guess]
        rank = word_info['rank']
        clue = word_info['clue'] if word_info['clue'].strip() else "Not a close association"
        
        # Check if it's the secret word
        if singular_guess == self.secret_word:
            print("\n" + "🎉" * 20)
            print("🏆 CONGRATULATIONS! YOU WON! 🏆")
            print("🎉" * 20)
            self.display_result(singular_guess, rank, clue, is_winner=True)
            return True
        
        # Add to guesses if not already guessed
        if not any(g[0] == singular_guess for g in self.guesses):
            self.guesses.append((singular_guess, rank, clue))
        
        # Display result
        self.display_result(singular_guess, rank, clue)
        
        # Display leaderboard
        self.display_leaderboard()
        
        return False  # Game continues
    
    def display_result(self, word: str, rank: int, clue: str, is_winner: bool = False):
        """Display the result of a guess"""
        print("\n" + "="*60)
        if is_winner:
            print("🎯 SECRET WORD FOUND! 🎯")
        else:
            print("📝 GUESS RESULT")
        print("="*60)
        print(f"Word: {word.upper()}")
        print(f"Rank: #{rank}")
        print(f"Clue: {clue}")
        print("="*60)
    
    def display_leaderboard(self):
        """Display the leaderboard of all guesses"""
        if not self.guesses:
            return
            
        print("\n🏆 LEADERBOARD (Best Guesses) 🏆")
        print("-" * 50)
        
        # Sort by rank (lower rank = better)
        sorted_guesses = sorted(self.guesses, key=lambda x: x[1])
        
        for i, (word, rank, clue) in enumerate(sorted_guesses, 1):
            print(f"{i:2d}. #{rank:3d} - {word.upper():15s} | {clue[:30]}...")
        
        print("-" * 50)
        print(f"Total guesses: {len(self.guesses)}")
        if sorted_guesses:
            print(f"Best rank so far: #{sorted_guesses[0][1]}")
        print()

def main():
    game = WordGameMaster()
    
    try:
        # Load data
        game.load_word_data('secretword/secretword-easy-animals-cat.csv')
        game.load_dictionary('data/enable2.txt')
        
        # Start game
        if not game.start_game():
            return
        
        # Game loop
        while True:
            try:
                guess = input("Enter your guess (or 'quit' to exit): ").strip()
                
                if guess.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for playing! 👋")
                    break
                
                if not guess:
                    continue
                
                # Process guess
                won = game.make_guess(guess)
                if won:
                    play_again = input("\nPlay again? (y/n): ").strip().lower()
                    if play_again in ['y', 'yes']:
                        game.start_game()
                    else:
                        print("Thanks for playing! 👋")
                        break
                        
            except KeyboardInterrupt:
                print("\n\nThanks for playing! 👋")
                break
                
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
