#!/usr/bin/env python3
"""
Mystery Word Game - Play without knowing the secret word
"""
import sys
import random
import os
from universal_game_master import create_game_master

def get_available_games():
    """Get list of available game words"""
    import glob
    
    # Look for CSV files in secretword directory
    csv_files = glob.glob("secretword/secretword-easy-animals-*.csv")
    games = []
    
    for file in csv_files:
        # Extract word from filename
        basename = os.path.basename(file)
        if basename.startswith("secretword-easy-animals-"):
            word = basename.replace("secretword-easy-animals-", "").replace(".csv", "")
            # Skip backup/temp files
            if not any(suffix in word for suffix in ["_backup", "_temp", "_incomplete"]):
                games.append(word)
    
    return sorted(games)

def show_game_prompt():
    """Display the game prompt"""
    prompt_file = "secretword/game-prompt.md"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove markdown headers for cleaner display
            lines = content.split('\n')
            for line in lines:
                if line.startswith('#'):
                    # Convert markdown headers to styled text
                    level = len(line) - len(line.lstrip('#'))
                    text = line.lstrip('# ').strip()
                    if level == 1:
                        print("\n" + "="*60)
                        print(f"  {text}")
                        print("="*60)
                    elif level == 2:
                        print(f"\n📋 {text}")
                        print("-" * 40)
                    elif level == 3:
                        print(f"\n🔹 {text}")
                else:
                    print(line)
    else:
        print("🎮 Welcome to the Universal Word Guessing Game!")
        print("Try to guess the secret word based on the clues you receive!")

def select_random_game():
    """Select a random game without revealing the word"""
    available_games = get_available_games()
    
    if not available_games:
        print("❌ No game files found!")
        return None
    
    # Select random word
    secret_word = random.choice(available_games)
    return secret_word

def play_mystery_game():
    """Play the game without knowing the secret word"""
    
    # Show the game prompt
    show_game_prompt()
    
    # Select random word
    secret_word = select_random_game()
    if not secret_word:
        return
    
    # Create game master (but don't reveal the word)
    game = create_game_master(secret_word)
    
    # Initialize the game (suppress word name in output)
    print("\n🔄 Loading mystery word data...")
    
    # Temporarily redirect stdout to hide the secret word in loading messages
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        success = game.initialize_game()
    finally:
        sys.stdout = old_stdout
    
    if not success:
        print("❌ Failed to initialize game!")
        return
    
    # Start the game with mystery theme
    game.game_started = True
    game.guesses = []
    game.hints_used = 0
    game.hint_history = []
    
    print("\n" + "="*60)
    print("🎭 MYSTERY WORD GUESSING GAME READY! 🎭")
    print("="*60)
    print("🎯 The secret word has been chosen...")
    print("🎲 Make your guesses and follow the clues!")
    print("📝 Each guess gets a rank and hint!")
    print("💡 Type 'hint' when you need help!")
    print("="*60)
    
    print(f"\n✅ Game loaded successfully!")
    print(f"📊 {len(game.word_data)} related words in database")
    print(f"🎲 Mystery word selected and ready!")
    
    while True:
        try:
            guess = input("\n🎯 Enter your guess (or 'quit' to exit): ").strip()
            
            if guess.lower() in ['quit', 'exit', 'q']:
                print(f"\n🎭 The secret word was: **{secret_word.upper()}**")
                print("Thanks for playing! 👋")
                break
            
            if not guess:
                continue
            
            # Process the guess
            result = game.process_guess(guess)
            
            # Handle winner reveal
            if result.get("winner"):
                print("\n" + "🎉" * 20)
                print("🏆 CONGRATULATIONS! YOU SOLVED THE MYSTERY! 🏆")
                print("🎉" * 20)
                print(f"\n🎭 The secret word was: **{secret_word.upper()}**")
                
                total_guesses = len(game.guesses)
                hints_used = game.hints_used
                
                print(f"\n📊 Final Stats:")
                print(f"   • Guesses made: {total_guesses}")
                print(f"   • Hints used: {hints_used}/{game.max_hints}")
                
                if hints_used == 0 and total_guesses <= 5:
                    print("🏆 PERFECT GAME! Amazing detective work!")
                elif hints_used == 0:
                    print("⭐ EXCELLENT! Solved without hints!")
                elif hints_used <= 3 and total_guesses <= 15:
                    print("🎯 GREAT JOB! Efficient solve!")
                elif total_guesses <= 20:
                    print("🎉 WELL DONE! Good detective work!")
                else:
                    print("🎉 MYSTERY SOLVED! Great persistence!")
                
                play_again = input("\nPlay another mystery? (y/n): ").strip().lower()
                if play_again in ['y', 'yes']:
                    return play_mystery_game()  # Start new mystery
                else:
                    print("Thanks for playing! 👋")
                    break
            else:
                # Display result normally (but keep word secret)
                game.display_result(result)
                
                # Show leaderboard for successful guesses
                if result.get("success") and not result.get("hint_level"):
                    game.display_leaderboard()
                elif result.get("hint_level"):
                    game.display_leaderboard()
                    
        except KeyboardInterrupt:
            print(f"\n\n🎭 The secret word was: **{secret_word.upper()}**")
            print("Thanks for playing! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point"""
    print("🎭 Welcome to the Mystery Word Game!")
    
    # Check if specific word requested via command line
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'random':
            play_mystery_game()
        else:
            # Allow specific word for testing
            secret_word = sys.argv[1].lower()
            print(f"🔧 Debug mode: Playing with '{secret_word}'")
            game = create_game_master(secret_word)
            if game.initialize_game():
                # Use the normal play interface but reveal word
                from play_universal_game import play_interactive_game
                play_interactive_game(secret_word)
            else:
                print(f"❌ Game '{secret_word}' not available!")
    else:
        play_mystery_game()

if __name__ == "__main__":
    main()
