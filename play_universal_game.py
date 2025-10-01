#!/usr/bin/env python3
"""
Universal Interactive Word Game Console
"""
import sys
from universal_game_master import create_game_master

def get_available_games():
    """Get list of available game words"""
    import os
    import glob
    
    # Look for CSV files in secretword directory
    csv_files = glob.glob("secretword/*-secret.csv")
    games = []
    
    for file in csv_files:
        # Extract word from filename
        basename = os.path.basename(file)
        if basename.endswith("-secret.csv"):
            word = basename.replace("-secret.csv", "")
            # Skip backup/temp files
            if not any(suffix in word for suffix in ["_backup", "_temp", "_incomplete"]):
                games.append(word)
    
    return sorted(games)

def select_game():
    """Let user select which game to play"""
    available_games = get_available_games()
    
    if not available_games:
        print("❌ No game files found in secretword directory!")
        return None
    
    print("🎮 Available Word Games:")
    print("-" * 40)
    for i, word in enumerate(available_games, 1):
        emoji = get_word_emoji(word)
        print(f"{i:2d}. {emoji} {word.upper()}")
    
    print("-" * 40)
    print("💡 Or type a word name directly!")
    
    while True:
        choice = input("\nSelect game (number or word name): ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_games):
                return available_games[idx]
            else:
                print(f"❌ Please enter a number between 1 and {len(available_games)}")
        else:
            word = choice.lower()
            if word in available_games:
                return word
            else:
                print(f"❌ '{choice}' not found. Available: {', '.join(available_games)}")

def get_word_emoji(word: str) -> str:
    """Get appropriate emoji for the word"""
    emoji_map = {
        'cat': '🐱', 'dog': '🐕', 'fish': '🐟', 'bird': '🐦',
        'queen': '👑', 'king': '👑', 'wine': '🍷', 'art': '🎨',
        'rock': '🪨', 'forest': '🌲', 'tree': '🌳', 'flower': '🌸',
        'ocean': '🌊', 'mountain': '⛰️', 'sun': '☀️', 'moon': '🌙'
    }
    return emoji_map.get(word, '🎮')

def get_word_hint(word: str) -> str:
    """Get a thematic hint for the word"""
    hints = {
        'cat': "A furry pet that purrs and catches mice",
        'dog': "Man's best friend with a wagging tail", 
        'fish': "Aquatic creature that swims and has gills",
        'bird': "Feathered creature that flies in the sky",
        'queen': "Royal female ruler with a crown",
        'king': "Male monarch who rules a kingdom",
        'wine': "Fermented grape beverage served in bottles",
        'art': "Creative expression through painting and sculpture",
        'rock': "Hard geological formation or mineral",
        'forest': "Dense area filled with trees and wildlife",
        'tree': "Tall woody plant with branches and leaves",
        'ocean': "Vast body of saltwater covering Earth",
        'mountain': "High elevated landform reaching toward sky"
    }
    return hints.get(word, "Try to guess the secret word!")

def play_interactive_game(secret_word: str):
    """Play the word guessing game interactively"""
    
    # Create game master
    game = create_game_master(secret_word)
    
    # Initialize the game
    if not game.initialize_game():
        print("❌ Failed to initialize game!")
        return
    
    game.start_game()
    
    print("🎯 Ready to play! Type your guesses:")
    print("🎲 Use your intuition and the clues to find the secret word!")
    print("💡 Type 'hint' when you need help!")
    
    while True:
        try:
            guess = input("\nEnter your guess (or 'quit' to exit): ").strip()
            
            if guess.lower() in ['quit', 'exit', 'q']:
                print("Thanks for playing! 👋")
                break
            
            if not guess:
                continue
            
            # Process the guess (including hints)
            result = game.process_guess(guess)
            game.display_result(result)
            
            # Show leaderboard for successful guesses (not for hints or errors)
            if result.get("success") and not result.get("winner") and not result.get("hint_level"):
                game.display_leaderboard()
            elif result.get("hint_level"):
                # For hints, show leaderboard after displaying hint
                game.display_leaderboard()
            
            # Handle winner
            if result.get("winner"):
                total_guesses = len(game.guesses)
                hints_used = game.hints_used
                
                print(f"\n🎉 Game Complete! 🎉")
                print(f"📊 Final Stats:")
                print(f"   • Guesses made: {total_guesses}")
                print(f"   • Hints used: {hints_used}/{game.max_hints}")
                
                if hints_used == 0:
                    print("🏆 Perfect game - no hints needed!")
                elif hints_used <= 2:
                    print("⭐ Great job with minimal hints!")
                
                play_again = input("\nPlay again? (y/n): ").strip().lower()
                if play_again in ['y', 'yes']:
                    # Ask if they want same word or different word
                    same_word = input(f"Play {secret_word.upper()} again? (y/n): ").strip().lower()
                    if same_word in ['y', 'yes']:
                        game.start_game()
                        print(f"🎯 Ready for another round of {secret_word.upper()}!")
                        print("🎲 Use your intuition and the clues to find the secret word!")
                        print("💡 Type 'hint' when you need help!")
                    else:
                        # Select new game
                        new_word = select_game()
                        if new_word:
                            return play_interactive_game(new_word)
                        else:
                            break
                else:
                    print("Thanks for playing! 👋")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nThanks for playing! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main game loop"""
    print("\n" + "="*60)
    print("🎮 UNIVERSAL WORD GUESSING GAME 🎮")
    print("="*60)
    
    # Check if word provided as command line argument
    if len(sys.argv) > 1:
        secret_word = sys.argv[1].lower().strip()
        print(f"🎯 Playing with word: {secret_word.upper()}")
    else:
        # Let user select game
        secret_word = select_game()
    
    if secret_word:
        play_interactive_game(secret_word)
    else:
        print("No game selected. Goodbye! 👋")

if __name__ == "__main__":
    main()




