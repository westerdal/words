#!/usr/bin/env python3
"""
Interactive Art Word Game Console
"""
from art_game_master import art_game

def play_interactive_art_game():
    """Play the art word guessing game interactively"""
    
    # Initialize the game
    if not art_game.initialize_game():
        print("❌ Failed to initialize game!")
        return
    
    art_game.start_game()
    
    print("🎯 Ready to play! Type your guesses:")
    print("🎨 Hint: The secret word is related to creativity and visual expression!")
    
    while True:
        try:
            guess = input("\nEnter your guess (or 'quit' to exit): ").strip()
            
            if guess.lower() in ['quit', 'exit', 'q']:
                print("Thanks for playing! 👋")
                break
            
            if not guess:
                continue
            
            # Process the guess
            won = False
            result = art_game.process_guess(guess)
            art_game.display_result(result)
            
            if result.get("success") and not result.get("winner"):
                art_game.display_leaderboard()
            
            if result.get("winner"):
                won = True
                
            if won:
                play_again = input("\nPlay again? (y/n): ").strip().lower()
                if play_again in ['y', 'yes']:
                    art_game.start_game()
                else:
                    print("Thanks for playing! 👋")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nThanks for playing! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    play_interactive_art_game()
